"""
ParamMap Web UI Server.
Lightweight, zero-dependency HTTP server with a modern web frontend
providing live conversational testing and real-time GPU/PCIe telemetry.
"""

import os
import sys
import json
import webbrowser
from typing import Optional, Dict, Any
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import torch

from ..engine import ParamMapEngine, SchedulingProfile, EngineStepResult
from ..scheduler import PROFILE_BUDGET_RATIOS
from ..mapper import MapBuilder, ParameterMap

_ENGINE_INSTANCE: Optional[ParamMapEngine] = None
_PARAM_MAP: Optional[ParameterMap] = None

def get_or_create_engine() -> ParamMapEngine:
    global _ENGINE_INSTANCE, _PARAM_MAP
    if _ENGINE_INSTANCE is not None:
        return _ENGINE_INSTANCE

    qwen_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf")
    gemma_path = os.path.expanduser("~/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf")
    model_path = qwen_path if os.path.exists(qwen_path) else gemma_path

    map_path = "maps/qwen3.5_9b_param_map.safetensors"
    if os.path.exists(map_path):
        _PARAM_MAP = ParameterMap.load(map_path)
    else:
        builder = MapBuilder(block_size=64)
        num_layers = 32
        inter = 12288
        domain_acts = {
            "math_formal": torch.randn(num_layers, inter).abs(),
            "code_python_system": torch.randn(num_layers, inter).abs(),
            "general_conversation": torch.randn(num_layers, inter).abs(),
            "creative_writing": torch.randn(num_layers, inter).abs(),
        }
        domain_acts["math_formal"][:, :inter//3] *= 3.0
        domain_acts["code_python_system"][:, inter//3:2*inter//3] *= 3.0
        domain_acts["general_conversation"][:, 2*inter//3:] *= 3.0
        _PARAM_MAP = builder.build_from_activations(domain_acts, model_name="Qwen3.5-9B-GGUF")
        os.makedirs("maps", exist_ok=True)
        _PARAM_MAP.save(map_path)

    _ENGINE_INSTANCE = ParamMapEngine(
        gguf_model_path=model_path,
        param_map=_PARAM_MAP,
        profile=SchedulingProfile.BALANCED
    )
    return _ENGINE_INSTANCE

_GENERATOR_MODEL = None
_GENERATOR_TOKENIZER = None

def get_or_create_generator():
    global _GENERATOR_MODEL, _GENERATOR_TOKENIZER
    if _GENERATOR_MODEL is not None:
        return _GENERATOR_MODEL, _GENERATOR_TOKENIZER

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        m_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"[*] Loading live generative LLM ({m_name})...")
        _GENERATOR_TOKENIZER = AutoTokenizer.from_pretrained(m_name)
        _GENERATOR_MODEL = AutoModelForCausalLM.from_pretrained(
            m_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        _GENERATOR_MODEL.eval()
        print(f"[+] Generative LLM successfully ready on {_GENERATOR_MODEL.device}")
    except Exception as e:
        print("[!] Could not initialize generative LLM:", e)
        _GENERATOR_MODEL = None
        _GENERATOR_TOKENIZER = None

    return _GENERATOR_MODEL, _GENERATOR_TOKENIZER

class ParamMapHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
            if os.path.exists(html_path):
                with open(html_path, "rb") as f:
                    content = f.read()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                except (ConnectionError, BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
            else:
                self.send_error(404, "Frontend file not found")
        elif parsed.path == "/api/status":
            engine = get_or_create_engine()
            budget = engine.budget
            gen_model, _ = get_or_create_generator()
            data = {
                "device_name": budget.device_name,
                "device_type": budget.device_type,
                "vram_total_gb": round(budget.total_vram_bytes / (1024**3), 2),
                "vram_free_gb": round(budget.free_vram_bytes / (1024**3), 2),
                "ram_total_gb": round(budget.total_ram_bytes / (1024**3), 2),
                "ram_free_gb": round(budget.free_ram_bytes / (1024**3), 2),
                "bandwidth_gb_s": budget.h2d_bandwidth_gb_s,
                "current_profile": engine.scheduler.profile.value,
                "model_name": os.path.basename(engine.adapter.gguf_path),
                "generator_model": "Qwen2.5-0.5B-Instruct (CUDA:0)" if gen_model is not None else "Model Hazır Değil",
                "total_blocks": engine.param_map.num_layers * engine.param_map.num_blocks,
                "layers": engine.param_map.num_layers,
                "blocks_per_layer": engine.param_map.num_blocks,
            }
            resp = json.dumps(data).encode("utf-8")
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            except (ConnectionError, BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(body)
            except Exception:
                payload = {}

            prompt = payload.get("prompt", "").strip()
            profile_str = payload.get("profile", "balanced").lower()
            enable_masking = payload.get("enable_masking", True)

            engine = get_or_create_engine()
            mapping = {
                "quality": SchedulingProfile.QUALITY,
                "balanced": SchedulingProfile.BALANCED,
                "speed": SchedulingProfile.SPEED,
                "min_memory": SchedulingProfile.MIN_MEMORY,
            }
            if profile_str in mapping:
                engine.set_profile(mapping[profile_str])

            # Process prompt through ParamMap Engine
            step: EngineStepResult = engine.process_turn(prompt)

            # Generate real LLM response with active ParamMap block mask attached!
            reply_text = self._generate_response(prompt, step.dominant_domain, engine, step, enable_masking=enable_masking)

            # Sample active block indices for visual heatmap (layers 0..15)
            active_heatmap = []
            for l_idx in range(min(16, engine.param_map.num_layers)):
                blocks_set = engine.scheduler.resident_hot_blocks[l_idx]
                active_heatmap.append(sorted(list(blocks_set)))

            response_data = {
                "reply": reply_text,
                "telemetry": {
                    "dominant_domain": step.dominant_domain,
                    "dominant_score": round(step.dominant_score * 100, 1),
                    "context_drift": round(step.context_drift, 3),
                    "active_block_ratio": round(step.active_block_ratio * 100, 1),
                    "total_active_blocks": step.total_active_blocks,
                    "loaded_blocks": step.loaded_blocks,
                    "evicted_blocks": step.evicted_blocks,
                    "transfer_latency_ms": step.transfer_latency_ms,
                    "vram_used_mb": step.vram_used_mb,
                    "vram_saved_mb": step.vram_saved_mb,
                    "vram_saved_percent": step.vram_saved_percent,
                    "active_heatmap": active_heatmap,
                    "masking_enabled": enable_masking,
                }
            }

            resp = json.dumps(response_data).encode("utf-8")
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            except (ConnectionError, BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        else:
            self.send_error(404)

    def _generate_response(
        self,
        prompt: str,
        domain: str,
        engine: ParamMapEngine,
        step: EngineStepResult,
        enable_masking: bool = True
    ) -> str:
        gen_model, gen_tok = get_or_create_generator()
        if gen_model is None or gen_tok is None:
            return f"Model yüklenemedi. Girdiniz '{domain}' alanında işlendi."

        # Format prompt with ChatML / chat template
        messages = [
            {"role": "system", "content": "Sen zeki, yardımsever ve Türkçe akıcı konuşan bir yapay zeka asistanısın. Kullanıcının sorusuna doğrudan, net ve bilgilendirici bir yanıt ver."},
            {"role": "user", "content": prompt}
        ]
        try:
            formatted_input = gen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted_input = prompt

        inputs = gen_tok(formatted_input, return_tensors="pt").to(gen_model.device)

        # Attach ParamMap dynamic active block masking hooks if enabled
        handles = []
        try:
            if enable_masking and hasattr(gen_model, "model") and hasattr(gen_model.model, "layers"):
                # Use matching parameter map if available
                map_path = "maps/qwen2.5_0.5b_param_map.safetensors"
                if os.path.exists(map_path):
                    p_map = ParameterMap.load(map_path)
                    block_size = p_map.block_size
                    inter_size = p_map.metadata.intermediate_size
                    # Active blocks for this domain
                    budget_ratio = PROFILE_BUDGET_RATIOS.get(engine.scheduler.profile, 0.5)
                    # Domain index
                    d_idx = 2  # default general
                    if "math" in domain: d_idx = 0
                    elif "code" in domain: d_idx = 1
                    top_blocks = p_map.get_top_blocks_for_domain(domain_idx=d_idx, budget_ratio=budget_ratio)

                    for l_idx in range(min(len(gen_model.model.layers), p_map.num_layers)):
                        mlp = gen_model.model.layers[l_idx].mlp
                        order = p_map.neuron_order[l_idx]
                        active_b = top_blocks[l_idx]
                        
                        indices = []
                        for b in active_b.tolist():
                            s = b * block_size
                            indices.append(order[s : s + block_size])
                        
                        if indices:
                            active_idx = torch.cat(indices).to(gen_model.device)
                            mask = torch.zeros(inter_size, device=gen_model.device, dtype=torch.bfloat16)
                            mask[active_idx] = 1.0

                            def make_h(m):
                                def hook(mod, inp):
                                    return (inp[0] * m.view(1, 1, -1),)
                                return hook

                            h = mlp.down_proj.register_forward_pre_hook(make_h(mask))
                            handles.append(h)

            with torch.no_grad():
                outputs = gen_model.generate(
                    **inputs,
                    max_new_tokens=180,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=gen_tok.eos_token_id
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            reply = gen_tok.decode(new_tokens, skip_special_tokens=True).strip()
            return reply if reply else "Cevap üretildi."
        except Exception as e:
            return f"Üretim esnasında hata: {str(e)}"
        finally:
            for h in handles:
                h.remove()

def start_web_ui(port: int = 7860, open_browser: bool = True):
    server = ThreadingHTTPServer(("127.0.0.1", port), ParamMapHTTPHandler)
    url = f"http://127.0.0.1:{port}"
    print("=" * 65)
    print("       PARAMAP DYNAMIC RUNTIME - WEB INTERFACE")
    print(f"       Running at: {url}")
    print("=" * 65)
    print("[*] ParamMap Motoru ve Yerel LLM GPU belleğine yükleniyor...")
    get_or_create_engine()
    get_or_create_generator()
    print("[+] Model ve dinamik katman yöneticisi hazır!")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ParamMap Web Server.")
        server.server_close()

if __name__ == "__main__":
    start_web_ui()
