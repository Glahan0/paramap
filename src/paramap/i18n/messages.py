"""
Internationalization (i18n) module for ParamMap supporting English and Turkish.
"""

from typing import Dict

DEFAULT_LANG = "en"

MESSAGES: Dict[str, Dict[str, str]] = {
    "device_detected": {
        "en": "Detected compute device: {device} ({details})",
        "tr": "Tespit edilen hesaplama aygıtı: {device} ({details})",
    },
    "vram_status": {
        "en": "VRAM: {free_gb:.2f} GB free / {total_gb:.2f} GB total",
        "tr": "VRAM: {free_gb:.2f} GB boş / {total_gb:.2f} GB toplam",
    },
    "ram_status": {
        "en": "System RAM: {free_gb:.2f} GB free / {total_gb:.2f} GB total",
        "tr": "Sistem RAM'i: {free_gb:.2f} GB boş / {total_gb:.2f} GB toplam",
    },
    "model_loading": {
        "en": "Loading model {model_name} in streaming/sharded mode...",
        "tr": "{model_name} modeli akış/bölümlü modda yükleniyor...",
    },
    "profile_layer_start": {
        "en": "Profiling layer {layer_idx}/{total_layers} across {num_domains} domains...",
        "tr": "Katman {layer_idx}/{total_layers}, {num_domains} alan üzerinde profilleniyor...",
    },
    "map_saved": {
        "en": "Parameter map successfully saved to: {path}",
        "tr": "Parametre haritası başarıyla kaydedildi: {path}",
    },
    "domain_discovery_start": {
        "en": "Discovering functional domains via dynamic LLM clustering...",
        "tr": "Dinamik LLM kümelemesi ile fonksiyonel alanlar keşfediliyor...",
    },
    "domain_signature_validated": {
        "en": "Validated domain '{domain_name}': Distinctiveness score {score:.3f}",
        "tr": "'{domain_name}' alanı doğrulandı: Ayırt edicilik puanı {score:.3f}",
    },
}

_current_lang = "en"

def set_language(lang: str) -> None:
    global _current_lang
    if lang.lower() in ("tr", "turkish", "turkce"):
        _current_lang = "tr"
    else:
        _current_lang = "en"

def get_language() -> str:
    return _current_lang

def tr(key: str, **kwargs) -> str:
    """Translates key to current language with formatting."""
    tmpl = MESSAGES.get(key, {}).get(_current_lang) or MESSAGES.get(key, {}).get("en", key)
    if kwargs:
        try:
            return tmpl.format(**kwargs)
        except KeyError:
            return tmpl
    return tmpl
