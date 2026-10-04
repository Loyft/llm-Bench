"""Config and model-ref helpers."""

from pathlib import Path

from llm_bench.config import merge_config
from llm_bench.providers.base import resolve_model_ref
from llm_bench.schema import ProviderName


def test_merge_config_cli_overrides(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "provider: ollama\nmodels: [a]\ntasks: [speed]\n",
        encoding="utf-8",
    )
    from llm_bench.config import load_yaml_config

    data = load_yaml_config(cfg_path)
    cfg = merge_config(data, models=["b", "c"], tools="off", tasks="all")
    assert cfg.models == ["b", "c"]
    assert cfg.tools is False
    assert "generation_speed" in cfg.tasks or "speed" in cfg.tasks or "latency" in cfg.tasks


def test_resolve_model_ref_path_and_url() -> None:
    url = resolve_model_ref("http://127.0.0.1:8080", ProviderName.LLAMACPP)
    assert url.is_url
    path = resolve_model_ref("./models/x.gguf", ProviderName.LLAMACPP)
    assert path.is_path
    name = resolve_model_ref("llama3.2:3b", ProviderName.OLLAMA)
    assert not name.is_path and not name.is_url
