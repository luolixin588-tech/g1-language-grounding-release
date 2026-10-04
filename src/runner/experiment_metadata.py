"""Collect run metadata used to prove experimental controls are held fixed."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

from src.paths import CATALOG_PATH, CONFIG_DIR, INSTRUCTIONS_PATH, REPO_ROOT

EVALUATOR_FILES = (
    REPO_ROOT / "src" / "evaluator" / "task_success.py",
    REPO_ROOT / "src" / "evaluator" / "plan_validator.py",
    REPO_ROOT / "src" / "evaluator" / "diagnostics.py",
    REPO_ROOT / "src" / "adapters" / "semantic_normalizer.py",
    REPO_ROOT / "src" / "adapters" / "production_plan_adapter.py",
    REPO_ROOT / "src" / "adapters" / "unstructured_plan_adapter.py",
    REPO_ROOT / "src" / "catalog.py",
    REPO_ROOT / "src" / "ablations" / "semantic_aliases.py",
    REPO_ROOT / "src" / "runner" / "planner_runner.py",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def load_model_config(model_id_or_tag: str) -> Dict[str, Any]:
    models_dir = CONFIG_DIR / "models"
    for path in sorted(models_dir.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if cfg.get("id") == model_id_or_tag or cfg.get("ollama_tag") == model_id_or_tag:
            cfg["_config_path"] = str(path)
            return cfg
    raise FileNotFoundError(f"no model config for {model_id_or_tag!r} in {models_dir}")


def list_model_configs() -> List[Dict[str, Any]]:
    models_dir = CONFIG_DIR / "models"
    out = []
    for path in sorted(models_dir.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg["_config_path"] = str(path)
        out.append(cfg)
    return out


def git_commit_hash() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or None
    except OSError:
        return None
    return None


def gpu_info() -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            return {"nvidia_smi": lines, "query_ok": True}
    except OSError:
        pass
    return {"nvidia_smi": [], "query_ok": False}


def ollama_version(base_url: str = "http://127.0.0.1:11434") -> Optional[str]:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/version", timeout=10)
        resp.raise_for_status()
        return str(resp.json().get("version") or "")
    except Exception:
        return None


def ollama_model_info(
    ollama_tag: str, base_url: str = "http://127.0.0.1:11434"
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "ollama_tag": ollama_tag,
        "present": False,
        "digest": None,
        "parameter_size": None,
        "quantization": None,
        "family": None,
        "modified_at": None,
        "size": None,
    }
    try:
        tags = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=15)
        tags.raise_for_status()
        for row in tags.json().get("models") or []:
            name = row.get("name") or ""
            model = row.get("model") or ""
            if name == ollama_tag or model == ollama_tag:
                details = row.get("details") or {}
                info.update(
                    {
                        "present": True,
                        "digest": row.get("digest"),
                        "parameter_size": details.get("parameter_size"),
                        "quantization": details.get("quantization_level"),
                        "family": details.get("family"),
                        "modified_at": row.get("modified_at"),
                        "size": row.get("size"),
                    }
                )
                break
    except Exception as exc:
        info["tags_error"] = str(exc)

    try:
        shown = requests.post(
            f"{base_url.rstrip('/')}/api/show",
            json={"name": ollama_tag},
            timeout=30,
        )
        if shown.ok:
            body = shown.json()
            details = body.get("details") or {}
            info["show_parameter_size"] = details.get("parameter_size")
            info["show_quantization"] = details.get("quantization_level")
            info["show_family"] = details.get("family")
            info["show_format"] = details.get("format")
            if not info.get("parameter_size"):
                info["parameter_size"] = details.get("parameter_size")
            if not info.get("quantization"):
                info["quantization"] = details.get("quantization_level")
    except Exception as exc:
        info["show_error"] = str(exc)
    return info


def protocol_hashes(prompt: str, schema: Optional[Dict[str, Any]]) -> Dict[str, str]:
    evaluator_blob = b"".join(path.read_bytes() for path in EVALUATOR_FILES)
    return {
        "prompt_sha256": sha256_text(prompt),
        "capability_catalog_sha256": sha256_file(CATALOG_PATH),
        "dataset_instructions_sha256": sha256_file(INSTRUCTIONS_PATH),
        "schema_sha256": sha256_json(schema) if schema is not None else "schema_disabled",
        "evaluator_sha256": sha256_bytes(evaluator_blob),
    }


def collect_run_metadata(
    *,
    model_display_name: str,
    ollama_tag: str,
    condition_name: str,
    condition: Dict[str, Any],
    prompt: str,
    schema: Optional[Dict[str, Any]],
    temperature: float,
    seed: int,
    repeats: int,
    smoke: bool,
    endpoint: str,
) -> Dict[str, Any]:
    base = endpoint.rsplit("/api/", 1)[0] if "/api/" in endpoint else "http://127.0.0.1:11434"
    model_info = ollama_model_info(ollama_tag, base_url=base)
    hashes = protocol_hashes(prompt, schema)
    return {
        "model_display_name": model_display_name,
        "ollama_tag": ollama_tag,
        "exact_model_digest": model_info.get("digest"),
        "parameter_size": model_info.get("parameter_size"),
        "quantization": model_info.get("quantization"),
        "ollama_version": ollama_version(base),
        "temperature": temperature,
        "seed": seed,
        "benchmark_version": "v1.1",
        "condition": condition_name,
        "repeats": repeats,
        "smoke": smoke,
        "gpu": gpu_info(),
        "git_commit_hash": git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_enabled": schema is not None,
        "endpoint": endpoint,
        "ollama_model_info": model_info,
        "condition_flags": {
            "use_json_schema": bool(condition.get("use_json_schema")),
            "use_validation_retries": bool(condition.get("use_validation_retries")),
            "max_attempts": condition.get("max_attempts"),
            "anonymize_capability_semantics": bool(
                condition.get("anonymize_capability_semantics")
            ),
            "anonymize_parameter_semantics": bool(
                condition.get("anonymize_parameter_semantics")
            ),
            "reduced_capability_info": bool(condition.get("reduced_capability_info")),
        },
        **hashes,
    }
