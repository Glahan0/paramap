"""
Loading a model without copying its weights into RAM.

`AutoModelForCausalLM.from_pretrained` reads every weight into private process
memory: a Qwen2.5-3B load grows the process by 5.8 GB, and a bf16 7B model
(~15 GB) does not fit a 16 GB machine at all. Yet most of those bytes are only
ever read: resident layers are copied to the GPU once, streamed and CPU layers
are read on every token.

So the weights are left in the safetensors files. Each file is memory-mapped
and every parameter becomes a zero-copy view into the mapping. The OS pages the
bytes in on first touch and keeps them in its file cache while RAM allows,
evicting them under pressure instead of failing -- which makes the disk a
third tier below VRAM and RAM:

    VRAM     resident layers and anchors (copied once at install)
    RAM      streamed layers that `StreamingModel` page-locks, and the file
             cache's copy of the rest
    disk     whatever the file cache had to give back; read again when needed

The output is unchanged: the same bytes reach the same kernels, they are just
read from a different place.
"""

import json
import mmap
import os
import struct
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# safetensors dtype tags -> torch dtypes.
_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


class SafetensorsMmap:
    """
    Zero-copy tensors over one or more memory-mapped safetensors files.

    The files are mapped read-only. A copy-on-write mapping would make the
    views writable, but Windows charges commit for the whole mapping up front
    (a 15 GB model would need 15 GB of commit it never uses). Nothing in
    inference writes to a weight; one that tried would fault, not corrupt the
    checkpoint.
    """

    def __init__(self, paths: List[str]):
        self.paths = list(paths)
        self._maps: List[mmap.mmap] = []
        self._index: Dict[str, Tuple[int, str, List[int], int, int]] = {}
        for path in self.paths:
            self._open(path)

    def _open(self, path: str) -> None:
        with open(path, "rb") as handle:
            (header_len,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(header_len))
            mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        file_idx = len(self._maps)
        self._maps.append(mapping)

        data_start = 8 + header_len
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            self._index[name] = (file_idx, info["dtype"], info["shape"], data_start + begin, end - begin)

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def keys(self) -> List[str]:
        return list(self._index)

    @property
    def total_bytes(self) -> int:
        return sum(entry[4] for entry in self._index.values())

    def tensor(self, name: str) -> torch.Tensor:
        file_idx, dtype_tag, shape, offset, nbytes = self._index[name]
        dtype = _DTYPES.get(dtype_tag)
        if dtype is None:
            raise ValueError(f"{name}: unsupported safetensors dtype {dtype_tag}")
        count = nbytes // torch.empty((), dtype=dtype).element_size()
        if count == 0:
            return torch.empty(shape, dtype=dtype)
        with warnings.catch_warnings():
            # torch warns that the buffer is read-only; that is the point.
            warnings.filterwarnings("ignore", message=".*not writable.*")
            flat = torch.frombuffer(self._maps[file_idx], dtype=dtype, count=count, offset=offset)
        return flat.view(shape)

    def release(self, tensor: torch.Tensor) -> None:
        """
        Drops a mapped tensor's pages from this process's working set once
        their bytes have been copied elsewhere (to the GPU, or into pinned RAM).

        They stay in the OS file cache, so touching them again is a cheap soft
        fault, but they count as available memory again. Without this, every
        byte copied out of the mapping keeps counting as used RAM, and loading
        a 3B model to the GPU left too little "available" to pin anything.
        A no-op for tensors that are not views into this store.
        """
        if tensor.device.type != "cpu" or not tensor.numel():
            return
        start = tensor.data_ptr()
        end = start + tensor.numel() * tensor.element_size()
        if not any(lo <= start and end <= hi for lo, hi in self._ranges()):
            return
        _drop_from_working_set(start, end - start)

    def _ranges(self) -> List[Tuple[int, int]]:
        if not hasattr(self, "_range_cache"):
            ranges = []
            for mapping in self._maps:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*not writable.*")
                    base = torch.frombuffer(mapping, dtype=torch.uint8, count=1).data_ptr()
                ranges.append((base, base + len(mapping)))
            self._range_cache = ranges
        return self._range_cache


_PAGE = mmap.PAGESIZE


def _drop_from_working_set(address: int, nbytes: int) -> None:
    import ctypes

    # Only whole pages inside the range, so neighbours are not disturbed.
    start = -(-address // _PAGE) * _PAGE
    end = (address + nbytes) // _PAGE * _PAGE
    if end <= start:
        return
    if os.name == "nt":
        # Documented: VirtualUnlock on pages that are not locked removes them
        # from the working set (and reports ERROR_NOT_LOCKED, ignored here).
        ctypes.windll.kernel32.VirtualUnlock(ctypes.c_void_p(start), ctypes.c_size_t(end - start))
    else:
        libc = ctypes.CDLL(None)
        libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(end - start), 4)  # MADV_DONTNEED


def resolve_checkpoint(model_name: str) -> str:
    """A local directory as-is, otherwise the Hub snapshot (downloaded if missing)."""
    if os.path.isdir(model_name):
        return model_name
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_name,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
    )


def safetensors_files(directory: str) -> List[str]:
    index_path = os.path.join(directory, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        return [os.path.join(directory, name) for name in sorted(set(weight_map.values()))]
    single = os.path.join(directory, "model.safetensors")
    return [single] if os.path.exists(single) else []


def _set_tensor(model: nn.Module, name: str, value: Any, is_param: bool) -> None:
    module_path, _, attr = name.rpartition(".")
    module = model.get_submodule(module_path) if module_path else model
    if is_param:
        module._parameters[attr] = value
    else:
        module._buffers[attr] = value


def load_model_mmap(
    model_name: str,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[nn.Module, str]:
    """
    Builds a causal LM whose parameters are views into its memory-mapped
    safetensors files. Returns (model, checkpoint directory).

    The model looks like an ordinary CPU model to everything downstream. Loading
    takes seconds whatever the model size and costs no private RAM; the bytes
    are read when something first touches them.

    A tensor stored in a dtype other than `dtype` has to be converted, and the
    converted copy does live in private RAM; that is reported, not hidden.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

    directory = resolve_checkpoint(model_name)
    files = safetensors_files(directory)
    if not files:
        raise FileNotFoundError(f"No safetensors weights in {directory}")
    store = SafetensorsMmap(files)

    config = AutoConfig.from_pretrained(directory)
    if dtype is None:
        dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None) or torch.float32
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

    # Parameters on the meta device (no memory); buffers such as rotary
    # frequencies are computed for real, since checkpoints rarely store them.
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)

    # First the tensors the checkpoint holds, then aliases (tied embeddings)
    # pointing at whatever their original parameter became.
    replaced: Dict[int, nn.Parameter] = {}
    converted: List[str] = []
    pending: List[Tuple[str, nn.Parameter]] = []
    for name, param in model.named_parameters(remove_duplicate=False):
        if id(param) in replaced:
            _set_tensor(model, name, replaced[id(param)], is_param=True)
            continue
        if name not in store:
            pending.append((name, param))
            continue
        tensor = store.tensor(name)
        if tuple(tensor.shape) != tuple(param.shape):
            raise ValueError(f"{name}: checkpoint shape {tuple(tensor.shape)} != model {tuple(param.shape)}")
        if tensor.dtype != param.dtype:
            tensor = tensor.to(param.dtype)
            converted.append(name)
        new_param = nn.Parameter(tensor, requires_grad=False)
        replaced[id(param)] = new_param
        _set_tensor(model, name, new_param, is_param=True)

    for name, param in pending:
        if id(param) in replaced:
            _set_tensor(model, name, replaced[id(param)], is_param=True)
    if pending:
        # Tied weights (e.g. lm_head sharing the embedding) are not stored;
        # the model knows which they are.
        model.tie_weights()
    missing = [name for name, param in model.named_parameters() if param.device.type == "meta"]
    if missing:
        raise ValueError(f"Checkpoint {directory} has no weights for: {', '.join(missing[:5])}")

    for name, buf in model.named_buffers():
        if name in store:
            _set_tensor(model, name, store.tensor(name).to(buf.dtype).clone(), is_param=False)

    if converted:
        print(
            f"Note: {len(converted)} tensors are stored in another dtype and were converted "
            f"into RAM (e.g. {converted[0]}); load with the checkpoint's dtype to avoid this."
        )

    try:
        model.generation_config = GenerationConfig.from_pretrained(directory)
    except (OSError, ValueError):
        pass

    model.eval()
    # Keeps the mappings alive as long as the model, and tells StreamingModel
    # that the original tensors cost no RAM to hold on to.
    model._paramap_store = store
    return model, directory


def is_file_backed(model: nn.Module) -> bool:
    return getattr(model, "_paramap_store", None) is not None


def release_file_pages(model: nn.Module, tensors) -> None:
    """`SafetensorsMmap.release` for each tensor; a no-op for ordinary models."""
    store = getattr(model, "_paramap_store", None)
    if store is None:
        return
    for tensor in tensors:
        store.release(tensor)
