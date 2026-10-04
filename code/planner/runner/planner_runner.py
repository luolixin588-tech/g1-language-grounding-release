from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from src.adapters.production_plan_adapter import parse_production_plan
from src.adapters.semantic_normalizer import normalize_steps
from src.adapters.unstructured_plan_adapter import parse_unstructured_plan
from src.evaluator.plan_validator import validate_production_then_semantic

# (planner_visible_steps) -> (decoded_steps, decode_errors)
ProductionDecoder = Callable[
    [Optional[List[Dict[str, Any]]]],
    Tuple[Optional[List[Dict[str, Any]]], List[str]],
]
FeedbackScrubber = Callable[[str], str]


class PlannerRunner:
    """Ollama planner. Validation retries are generation recovery, not execution-time replanning."""

    def __init__(
        self,
        ollama_url: str,
        model: str,
        planner_prompt: str,
        timeout_s: int = 90,
        max_transport_retries: int = 3,
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.planner_prompt = planner_prompt
        self.timeout_s = timeout_s
        self.max_transport_retries = max_transport_retries

    def _request(
        self,
        user_prompt: str,
        tool_catalog: List[Dict[str, Any]],
        json_schema: Optional[Dict[str, Any]],
        feedback: str,
        extra_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self.planner_prompt},
            {
                "role": "system",
                "content": (
                    "Registered robot capabilities (name/description/parameters). "
                    "Use only these tools and parameter values:\n"
                    + json.dumps(tool_catalog, ensure_ascii=False)
                ),
            },
        ]
        if extra_messages:
            messages.extend(extra_messages)
        if feedback:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "The previous plan was not executable under the catalog. "
                        f"Fix it. Error: {feedback}"
                    ),
                }
            )
        messages.append({"role": "user", "content": user_prompt})
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0},
        }
        if json_schema is not None:
            payload["format"] = json_schema
        return self._post_with_transport_retry(payload)

    def _post_with_transport_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Retry only transport/Ollama failures. Request payload is never modified."""
        last_exc: Optional[Exception] = None
        attempts = 1 + max(0, int(self.max_transport_retries))
        for try_i in range(attempts):
            try:
                resp = requests.post(self.ollama_url, json=payload, timeout=self.timeout_s)
                if 500 <= resp.status_code < 600:
                    raise requests.HTTPError(
                        f"{resp.status_code} Server Error", response=resp
                    )
                resp.raise_for_status()
                body = resp.json()
                body["_transport_retries"] = try_i
                return body
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                if try_i >= attempts - 1:
                    break
                time.sleep(1.0 * (try_i + 1))
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if 500 <= status < 600:
                    last_exc = exc
                    if try_i >= attempts - 1:
                        break
                    time.sleep(1.0 * (try_i + 1))
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def plan(
        self,
        user_prompt: str,
        tool_catalog: List[Dict[str, Any]],
        json_schema: Optional[Dict[str, Any]],
        max_attempts: int,
        use_validation_retries: bool,
        extra_messages: Optional[List[Dict[str, str]]] = None,
        production_decoder: Optional[ProductionDecoder] = None,
        feedback_scrubber: Optional[FeedbackScrubber] = None,
    ) -> Dict[str, Any]:
        attempt_latencies: List[float] = []
        attempt_errors: List[str] = []
        attempt_ollama_timings: List[Dict[str, Any]] = []
        transport_retries = 0
        last_raw = ""
        last_visible = None
        last_decoded = None
        last_semantic = None
        last_validity = {"valid": False, "errors": ["no attempt"]}
        feedback = ""
        t_all = time.perf_counter()

        for attempt in range(1, max_attempts + 1):
            t0 = time.perf_counter()
            try:
                body = self._request(
                    user_prompt, tool_catalog, json_schema, feedback, extra_messages
                )
                transport_retries += int(body.get("_transport_retries") or 0)
                attempt_ollama_timings.append(
                    {
                        "load_duration": body.get("load_duration"),
                        "prompt_eval_duration": body.get("prompt_eval_duration"),
                        "eval_duration": body.get("eval_duration"),
                        "total_duration": body.get("total_duration"),
                        "eval_count": body.get("eval_count"),
                        "prompt_eval_count": body.get("prompt_eval_count"),
                        "transport_retries": int(body.get("_transport_retries") or 0),
                    }
                )
                content = (body.get("message", {}) or {}).get("content") or ""
                last_raw = content if isinstance(content, str) else json.dumps(content)
                parser = (
                    parse_unstructured_plan if json_schema is None else parse_production_plan
                )
                visible, parse_errors = parser(last_raw)
                last_visible = visible
                decode_errors: List[str] = []
                decoded = visible
                if visible is not None and production_decoder is not None:
                    decoded, decode_errors = production_decoder(visible)
                last_decoded = decoded
                if decoded is None:
                    semantic, norm_errors = None, parse_errors + decode_errors
                else:
                    semantic, norm_errors = normalize_steps(decoded)
                last_semantic = semantic
                last_validity = validate_production_then_semantic(
                    decoded, semantic, parse_errors + decode_errors + norm_errors
                )
                elapsed = (time.perf_counter() - t0) * 1000
                attempt_latencies.append(elapsed)
                if last_validity["valid"]:
                    break
                err = "; ".join(last_validity["errors"]) or "invalid plan"
                attempt_errors.append(f"attempt {attempt}: {err}")
                if not use_validation_retries:
                    break
                feedback = err
                if feedback_scrubber is not None:
                    feedback = feedback_scrubber(feedback)
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                attempt_latencies.append(elapsed)
                last_raw = last_raw or ""
                last_validity = {"valid": False, "errors": [f"planner request failed: {e}"]}
                attempt_errors.append(f"attempt {attempt}: {e}")
                if not use_validation_retries:
                    break
                feedback = str(e)
                if feedback_scrubber is not None:
                    feedback = feedback_scrubber(feedback)

        return {
            "raw_model_output": last_raw,
            "parsed_production_plan": last_visible,
            "decoded_production_plan": last_decoded,
            "normalized_semantic_plan": last_semantic,
            "validity": last_validity,
            "attempt_count": len(attempt_latencies),
            "validation_errors": last_validity.get("errors") or [],
            "attempt_errors": attempt_errors,
            "latency_ms_per_attempt": attempt_latencies,
            "total_planning_latency_ms": (time.perf_counter() - t_all) * 1000,
            "ollama_timings": attempt_ollama_timings,
            "transport_retries": transport_retries,
        }
