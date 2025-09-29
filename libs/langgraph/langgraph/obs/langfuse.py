from __future__ import annotations

import base64
import json
import os
import time
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import requests
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

LANGFUSE_ENABLED_ENV = "LANGFUSE_ENABLED"
LANGFUSE_HOST_ENV = "LANGFUSE_HOST"
LANGFUSE_DSN_ENV = "LANGFUSE_DSN"  # Optional DSN: http://PUBLIC:SECRET@host:port
LANGFUSE_PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
LANGFUSE_SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"
LANGFUSE_INGESTION_PATH = "/api/public/ingestion"
LANGFUSE_TRACE_MODE_ENV = "LANGFUSE_TRACE_MODE"  # "graph" (default) | "llm"


def _basic_auth_header(public_key: str, secret_key: str) -> str:
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
    return f"Basic {token}"


def _now_iso() -> str:
    # RFC3339 / ISO8601 with UTC timezone suffix expected by Langfuse
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class _IngestionClient:
    """Very small HTTP client for Langfuse ingestion.

    Intentionally synchronous and minimal to avoid adding deps.
    """

    def __init__(self, host: str, public_key: str, secret_key: str) -> None:
        self.url = host.rstrip("/") + LANGFUSE_INGESTION_PATH
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": _basic_auth_header(public_key, secret_key),
        }

    def post_events(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        try:
            payload = {"batch": events}
            resp = requests.post(
                self.url, headers=self.headers, data=json.dumps(payload), timeout=5
            )
            if os.getenv("LANGFUSE_DEBUG", "").lower() in {
                "1",
                "true",
                "yes",
                "verbose",
            }:
                try:
                    print(
                        f"[Langfuse] POST {self.url} status={resp.status_code} events={len(events)}",
                        flush=True,
                    )
                    if (
                        resp.status_code not in (200, 201)
                        or os.getenv("LANGFUSE_DEBUG", "").lower() == "verbose"
                    ):
                        print(
                            f"[Langfuse] response: {resp.text[:1000]}",
                            flush=True,
                        )
                    if os.getenv("LANGFUSE_DEBUG", "").lower() == "verbose":
                        print(
                            f"[Langfuse] payload: {json.dumps(payload)[:1000]}",
                            flush=True,
                        )
                except Exception:
                    pass
        except Exception:
            # Best-effort; do not raise to avoid impacting user runs
            if os.getenv("LANGFUSE_DEBUG", "").lower() in {"1", "true", "yes"}:
                try:
                    print("[Langfuse] Failed to POST events", flush=True)
                except Exception:
                    pass
            pass


class LangfuseIngestionHandler(BaseCallbackHandler):
    """Minimal callback handler that forwards events to Langfuse ingestion API.

    This is intentionally minimal (create-only, best-effort, no retries/batching).
    """

    run_inline = True

    def __init__(self, host: str, public_key: str, secret_key: str) -> None:
        self._client = _IngestionClient(host, public_key, secret_key)
        self._mode = os.getenv(LANGFUSE_TRACE_MODE_ENV, "graph").lower()
        # Per LLM run state (only used in LLM mode)
        self._llm_state: dict[str, dict[str, Any]] = {}

    # --- helpers ---------------------------------------------------------
    def _event(self, type_: str, body: dict[str, Any]) -> dict[str, Any]:
        # Minimal event payload for ingestion API
        return {
            "type": type_,
            "id": body.get("id") or body.get("run_id") or body.get("trace_id"),
            "timestamp": _now_iso(),
            "body": body,
        }

    def _trace_body(
        self, run_id: UUID, name: str | None, metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        return {
            "id": str(run_id),
            "name": name or "run",
            "metadata": metadata or {},
        }

    def _span_body(
        self, run_id: UUID, name: str | None, parent_run_id: UUID | None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "id": str(run_id),
            "name": name or "span",
        }
        if parent_run_id is not None:
            body["parentId"] = str(parent_run_id)
        return body

    def _extract_chat_messages(
        self, messages: list[list[BaseMessage]]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for group in messages:
            for m in group:
                role = getattr(m, "type", None) or getattr(m, "role", None) or "user"
                content = getattr(m, "content", None)
                out.append({"role": role, "content": content})
        return out

    def _pick_output_text(self, response: Any) -> dict[str, Any]:
        try:
            gens = getattr(response, "generations", None)
            if gens and gens[0]:
                gen0 = gens[0][0]
                text = getattr(gen0, "text", None)
                if text is not None:
                    return {"text": text}
                message = getattr(gen0, "message", None)
                if message is not None:
                    return {"text": getattr(message, "content", None)}
        except Exception:
            pass
        return {"text": None}

    def _extract_usage_and_model(
        self, response: Any, serialized: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None]:
        usage: dict[str, Any] = {}
        model: str | None = None
        try:
            llm_output = getattr(response, "llm_output", None) or {}
            token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
            prompt = token_usage.get("prompt_tokens") or token_usage.get("promptTokens")
            completion = token_usage.get("completion_tokens") or token_usage.get(
                "completionTokens"
            )
            total = token_usage.get("total_tokens") or token_usage.get("totalTokens")
            if any(v is not None for v in (prompt, completion, total)):
                usage = {
                    "promptTokens": prompt,
                    "completionTokens": completion,
                    "totalTokens": total,
                }
            model = llm_output.get("model_name") or llm_output.get("model")
            if not model:
                model = (serialized.get("kwargs") or {}).get("model") or (
                    serialized.get("kwargs") or {}
                ).get("model_name")
        except Exception:
            pass
        return usage, model

    # --- chain (graph/node) events --------------------------------------
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._mode == "graph":
            body = self._trace_body(run_id, name, metadata)
            try:
                if isinstance(inputs, dict) and inputs:
                    body["input"] = inputs
            except Exception:
                pass
            events = [
                self._event("trace-create", body),
                self._event(
                    "span-create", self._span_body(run_id, name, parent_run_id)
                ),
            ]
            self._client.post_events(events)

    def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._mode == "graph":
            event = self._event(
                "event-create",
                {
                    "id": str(run_id),
                    "name": "chain_end",
                    "status": "success",
                    "output": outputs,
                },
            )
            self._client.post_events([event])

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        event = self._event(
            "event-create",
            {
                "id": str(run_id),
                "name": "chain_error",
                "status": "error",
                "error": repr(error),
            },
        )
        self._client.post_events([event])

    # --- LLM events ------------------------------------------------------
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str] | list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._mode == "graph":
            event = self._event(
                "span-create", self._span_body(run_id, name or "llm", parent_run_id)
            )
            self._client.post_events([event])
        else:
            try:
                if prompts and isinstance(prompts[0], list):
                    inp: Any = {"prompts": ["".join(map(str, p)) for p in prompts]}
                else:
                    inp = {"prompts": prompts}
            except Exception:
                inp = {"prompts": prompts}
            self._llm_state[str(run_id)] = {
                "start_ts": time.time(),
                "input": inp,
                "serialized": serialized,
                "tags": tags or [],
            }

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._mode == "llm":
            self._llm_state[str(run_id)] = {
                "start_ts": time.time(),
                "input": {"messages": self._extract_chat_messages(messages)},
                "serialized": serialized,
                "tags": tags or [],
            }

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._mode == "graph":
            event = self._event(
                "event-create",
                {"id": str(run_id), "name": "llm_end", "status": "success"},
            )
            self._client.post_events([event])
            return
        key = str(run_id)
        state = self._llm_state.pop(key, None) or {}
        start_ts = state.get("start_ts")
        latency_ms = int((time.time() - start_ts) * 1000) if start_ts else None
        serialized = state.get("serialized", {})
        usage, model = self._extract_usage_and_model(response, serialized)
        output = self._pick_output_text(response)
        input_payload = state.get("input")
        tags = state.get("tags") or []

        trace_id = str(run_id)
        trace_body = {
            "id": trace_id,
            "name": "llm-completion",
            "input": input_payload,
            "output": output,
            "metadata": {"model": model, "latency_ms": latency_ms},
            "tags": tags,
        }
        gen_id = f"gen-{trace_id}"
        gen_body = {
            "id": gen_id,
            "traceId": trace_id,
            "name": "llm-step",
            "model": model,
            "input": input_payload,
            "output": output,
            "usage": usage or None,
            "metadata": {"latency_ms": latency_ms},
            "tags": tags,
        }
        events = [
            self._event("trace-create", trace_body),
            self._event("generation-create", gen_body),
        ]
        self._client.post_events(events)

    def on_chat_model_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._mode == "graph":
            event = self._event(
                "event-create",
                {"id": str(run_id), "name": "chat_model_end", "status": "success"},
            )
            self._client.post_events([event])
            return
        key = str(run_id)
        state = self._llm_state.pop(key, None) or {}
        start_ts = state.get("start_ts")
        latency_ms = int((time.time() - start_ts) * 1000) if start_ts else None
        serialized = state.get("serialized", {})
        usage, model = self._extract_usage_and_model(response, serialized)
        output = self._pick_output_text(response)
        input_payload = state.get("input")
        tags = state.get("tags") or []

        trace_id = str(run_id)
        trace_body = {
            "id": trace_id,
            "name": "llm-completion",
            "input": input_payload,
            "output": output,
            "metadata": {"model": model, "latency_ms": latency_ms},
            "tags": tags,
        }
        gen_id = f"gen-{trace_id}"
        gen_body = {
            "id": gen_id,
            "traceId": trace_id,
            "name": "llm-step",
            "model": model,
            "input": input_payload,
            "output": output,
            "usage": usage or None,
            "metadata": {"latency_ms": latency_ms},
            "tags": tags,
        }
        events = [
            self._event("trace-create", trace_body),
            self._event("generation-create", gen_body),
        ]
        self._client.post_events(events)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._mode == "llm":
            key = str(run_id)
            state = self._llm_state.pop(key, None) or {}
            start_ts = state.get("start_ts")
            latency_ms = int((time.time() - start_ts) * 1000) if start_ts else None
            input_payload = state.get("input")
            tags = state.get("tags") or []
            trace_id = str(run_id)
            trace_body = {
                "id": trace_id,
                "name": "llm-completion-error",
                "input": input_payload,
                "output": {"error": repr(error)},
                "metadata": {"latency_ms": latency_ms, "error": True},
                "tags": tags,
            }
            err_event = self._event(
                "event-create",
                {
                    "id": trace_id,
                    "name": "llm_error",
                    "status": "error",
                    "error": repr(error),
                },
            )
            self._client.post_events(
                [
                    self._event("trace-create", trace_body),
                    err_event,
                ]
            )
            return
        event = self._event(
            "event-create",
            {
                "id": str(run_id),
                "name": "llm_error",
                "status": "error",
                "error": repr(error),
            },
        )
        self._client.post_events([event])

    def on_chat_model_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._mode == "llm":
            key = str(run_id)
            state = self._llm_state.pop(key, None) or {}
            start_ts = state.get("start_ts")
            latency_ms = int((time.time() - start_ts) * 1000) if start_ts else None
            input_payload = state.get("input")
            tags = state.get("tags") or []
            trace_id = str(run_id)
            trace_body = {
                "id": trace_id,
                "name": "llm-completion-error",
                "input": input_payload,
                "output": {"error": repr(error)},
                "metadata": {"latency_ms": latency_ms, "error": True},
                "tags": tags,
            }
            err_event = self._event(
                "event-create",
                {
                    "id": trace_id,
                    "name": "chat_model_error",
                    "status": "error",
                    "error": repr(error),
                },
            )
            self._client.post_events(
                [
                    self._event("trace-create", trace_body),
                    err_event,
                ]
            )
            return
        event = self._event(
            "event-create",
            {
                "id": str(run_id),
                "name": "chat_model_error",
                "status": "error",
                "error": repr(error),
            },
        )
        self._client.post_events([event])


def make_langfuse_handler_from_env() -> LangfuseIngestionHandler | None:
    """Create a handler if env vars indicate Langfuse should be enabled.

    Returns None if not enabled or misconfigured.
    """
    if os.getenv(LANGFUSE_ENABLED_ENV, "").lower() not in {"1", "true", "yes"}:
        return None
    # Support DSN form: http://PUBLIC:SECRET@host:port
    dsn = os.getenv(LANGFUSE_DSN_ENV)
    if dsn:
        try:
            parsed = urlparse(dsn)
            host = f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 80}"
            public_key = parsed.username or ""
            secret_key = parsed.password or ""
        except Exception:
            host = public_key = secret_key = None  # type: ignore
    else:
        host = os.getenv(LANGFUSE_HOST_ENV)
        public_key = os.getenv(LANGFUSE_PUBLIC_KEY_ENV)
        secret_key = os.getenv(LANGFUSE_SECRET_KEY_ENV)
    if not host or not public_key or not secret_key:
        return None
    try:
        return LangfuseIngestionHandler(host, public_key, secret_key)
    except Exception:
        return None
