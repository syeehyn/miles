"""Engine checks for routed-experts forcing on one GPU node. Test-only; see README.md.

Starts SGLang with --enable-routed-experts-forcing in three layouts (radix cache on, radix cache off, and, for
chat_session only, the ON arm's engine flags) and uses a copy of the checkpoint with perturbed MoE routers as the
updated weights:

- session: after a weight update, a session's second turn is admitted and forces every held row it recomputes; with the
  radix cache on, forced rows read back from the cache equal the held ones; the same turn without a key does change
  routes (otherwise the perturbation proves nothing); a key the engine never held is refused before compute with
  "routed_experts_missing".
- retract: a request retracted mid-decode keeps the rows it computed before the retract.
- chat_session: Miles' session server sends chat turns straight to SGLang and through the SGLang router, and every
  turn carries forcing counters with unforced_tokens 0.

Rows are compared as sets of experts. Use a small (for example 4-layer) checkpoint: the perturbed copy rewrites every
safetensors file that holds a router. Results go to --out-dir: engine-smoke.json, and per layout the SGLang and
router logs and chat-session-<layout>.json. Run from the repository root:

    python -m examples.experimental.routing_replay_forcing_ab.engine_smoke \\
        --model-path /root/models/GLM-5.3-Flash-4layer --tito-model glm53 \\
        --engine-args="--tp-size 4 --ep-size 4 --moe-runner-backend triton --mem-fraction-static 0.7 \\
            --chunked-prefill-size 8192 --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \\
            --kv-cache-dtype bfloat16"

or `python -m pytest examples/experimental/routing_replay_forcing_ab/engine_smoke.py` with ENGINE_SMOKE_MODEL_PATH,
ENGINE_SMOKE_TITO_MODEL and ENGINE_SMOKE_ENGINE_ARGS set.
"""

import argparse
import asyncio
import base64
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import requests
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tests.e2e.sglang.utils.sglang_server import start_sglang_server

from miles.utils.http_utils import find_available_port

_FORCING_ARGS = ("--enable-return-routed-experts", "--enable-routed-experts-forcing")
_NO_RADIX_ARG = "--disable-radix-cache"
# Where the engines of run_ab.py --arm on differ from the no-radix layout (its store holds 256 sessions x 32K tokens).
_RL_ENGINE_ARGS = (
    _NO_RADIX_ARG,
    "--enable-metrics",
    "--skip-server-warmup",
    "--random-seed",
    "1234",
    "--routed-experts-forcing-max-tokens",
    str(256 * 32768),
)
_ALL_CHECKS = ("chat_session", "session", "retract")
_LAYOUTS = (
    ("radix", (), _ALL_CHECKS),
    ("no-radix", (_NO_RADIX_ARG,), _ALL_CHECKS),
    ("rl-engine", _RL_ENGINE_ARGS, ("chat_session",)),
)
_PROMPT = (
    "Write a short, careful explanation of why the sum of the first n odd numbers equals n squared, "
    "and then check the claim for n = 1, 2, 3 and 4."
)
_TOOL_REPLY = "Tool result: the check passed for every n you listed. Continue."
_FILLER_LINE = "Record {i}: the meter read {a} at station {b}, and the log noted {c} units carried over."
_CONTEXT_TOKENS = 4096
_RETRACT_TOKENS = 2048
_RETRACT_DELAY_SECS = 1.0
_ROUTER_SUFFIX = ".mlp.gate.weight"
_PERTURB_SCALE = 0.2
# A w1 re-prefill without forcing that flips fewer rows than this cannot tell forcing from a no-op.
_MIN_CONTROL_FLIP_FRAC = 0.01
_STARTUP_TIMEOUT_SECS = 40 * 60
# Bounds every request, the retracted one included, on the 4-layer checkpoints this smoke runs.
_HTTP_TIMEOUT_SECS = 600


def run(
    *, model_path: str, engine_args: list[str], out_dir: str, tito_model: str, base_port: int, work_dir: str
) -> dict:
    """Every layout runs even when an earlier one fails; each result says whether it passed."""
    model = Path(model_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    perturbed = _perturbed_checkpoint(model, Path(work_dir) / f"{model.name}-perturbed-routers")
    results = []
    for layout, layout_args, checks in _LAYOUTS:
        try:
            outcome = _check_layout(
                model,
                perturbed,
                [*engine_args, *_FORCING_ARGS, *layout_args],
                out,
                layout=layout,
                checks=checks,
                tito_model=tito_model,
                base_port=base_port,
            )
            failed = [name for name in checks if not outcome[name]["passed"]]
            result = {"layout": layout, "passed": not failed, **outcome}
            if failed:
                result["error"] = f"failed checks: {', '.join(failed)}"
            results.append(result)
        except Exception as error:
            results.append({"layout": layout, "passed": False, "error": f"{type(error).__name__}: {error}"})
    evidence = {"model": model_path, "engine_args": engine_args, "layouts": results}
    (out / "engine-smoke.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


# Engine ports start here; the router's are 2000 above, its metrics 3000 above and the capture proxy 4000 above.
_DEFAULT_BASE_PORT = 21000
_SGLANG_ROUTER_PORT_OFFSET = 2000


def _check_layout(
    model: Path,
    perturbed: Path,
    server_args: list[str],
    run_dir: Path,
    *,
    layout: str,
    checks: tuple[str, ...],
    tito_model: str,
    base_port: int,
) -> dict:
    log_copy = run_dir / f"engine-smoke-{layout}.log"
    port = find_available_port(base_port)
    # start_sglang_server's log, which it leaves behind when the server fails to start
    log_path = Path(f"/tmp/sglang_e2e_{port}.log")
    server = None
    try:
        server = start_sglang_server(
            model_path=str(model),
            port=port,
            startup_timeout_secs=_STARTUP_TIMEOUT_SECS,
            # Deterministic inference would replace the attention backends these models' launchers use.
            enable_deterministic_inference=False,
            extra_args=server_args,
        )
        config = json.loads((model / "config.json").read_text())
        config = config.get("text_config", config)
        shape = (config["num_hidden_layers"], config["num_experts_per_tok"])
        capability = requests.get(f"{server.base_url}/server_info", timeout=60).json().get("routed_experts_forcing")
        assert (
            capability is not None and capability["version"] == 1
        ), f"no routed_experts_forcing capability: {capability}"
        radix = _NO_RADIX_ARG not in server_args
        runners = {
            "chat_session": lambda: _check_chat_session(
                server.base_url,
                shape,
                model,
                tito_model,
                run_dir,
                layout,
                base_port=base_port + _SGLANG_ROUTER_PORT_OFFSET,
            ),
            "session": lambda: _check_session(server.base_url, shape, model, perturbed, radix=radix),
            "retract": lambda: _check_retract(server.base_url, shape, model, perturbed),
        }
        outcome = {"capability": capability}
        for name in checks:
            try:
                result = runners[name]()
                outcome[name] = result if "passed" in result else {"passed": True, **result}
            except Exception as error:
                outcome[name] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
        return outcome
    finally:
        if server is not None:
            server.stop()
        if log_path.exists():
            shutil.copyfile(log_path, log_copy)


def _check_session(url: str, shape: tuple[int, int], model: Path, perturbed: Path, *, radix: bool) -> dict:
    key = f"forcing-smoke-{uuid.uuid4().hex}"
    prompt_ids = _long_prompt_ids(url)
    turn1 = _completed(_generate(url, prompt_ids, 32, session_key=key))
    seq1 = prompt_ids + _output_ids(turn1)
    held = len(seq1) - 1
    rows1 = _routes(turn1, shape)
    assert rows1.shape[0] == held, (rows1.shape, held)

    _update_weights(url, perturbed)
    try:
        turn2_ids = seq1 + _tokenize(url, _TOOL_REPLY, add_special_tokens=False)
        forced = _completed(_generate(url, turn2_ids, 8, session_key=key, start_len=held))
        # Turn 2 returns only the rows past the held ones. Its whole sequence, under its radix namespace, hits the KV
        # turn 2 just computed, whose captured rows are the forced ones.
        probe = None
        if radix:
            probe_ids = turn2_ids + _output_ids(forced)
            probe = _completed(_generate(url, probe_ids, 1, extra_key=f"routed_experts_session:{key}"))
        control = _completed(_generate(url, turn2_ids, 8))
        refused = _generate(url, turn2_ids, 8, session_key=f"{key}-never-held", start_len=held)
    finally:
        _update_weights(url, model)

    stats = forced["meta_info"]["routed_experts_forcing"]
    assert stats["held_tokens"] == held and stats["unforced_tokens"] == 0, stats
    assert stats["forced_tokens"] >= held - forced["meta_info"].get("cached_tokens", 0), stats
    result = {"held_tokens": held, "forced": stats}
    if probe is not None:
        # A linear-attention cache keeps its state only where requests ended, so it may serve fewer positions.
        read_back = min(probe["meta_info"]["cached_tokens"], held)
        differing = _differing_row_frac(_routes(probe, shape)[:read_back], rows1[:read_back]) if read_back else 0.0
        assert differing == 0, f"{differing:.4f} of the {read_back} forced rows read back differ from the held rows"
        result["read_back_tokens"] = read_back
    flips = _differing_row_frac(_routes(control, shape)[:held], rows1)
    assert flips > _MIN_CONTROL_FLIP_FRAC, f"the perturbation flips only {flips:.4f} of rows; raise _PERTURB_SCALE"
    refusal = _refusal(refused)
    want = {"code": "routed_experts_missing", "message": f"routed_experts_missing: held=0 requested={held}"}
    assert refusal == want, refusal
    return {**result, "control_flip_frac": flips, "refusal": refusal}


def _check_retract(url: str, shape: tuple[int, int], model: Path, perturbed: Path) -> dict:
    prompt_ids = _tokenize(url, _PROMPT, add_special_tokens=True)
    references = [_routes(_completed(_generate(url, prompt_ids, _RETRACT_TOKENS)), shape) for _ in range(2)]
    responses = []
    worker = threading.Thread(
        target=lambda: responses.append(
            _generate(url, prompt_ids, _RETRACT_TOKENS, session_key=f"forcing-smoke-retract-{uuid.uuid4().hex}")
        )
    )
    worker.start()
    time.sleep(_RETRACT_DELAY_SECS)
    _post(url, "pause_generation", {"mode": "retract"})
    _update_weights(url, perturbed)
    _post(url, "continue_generation", {})
    worker.join(timeout=_HTTP_TIMEOUT_SECS)
    _update_weights(url, model)

    assert not worker.is_alive(), f"the retracted request did not finish within {_HTTP_TIMEOUT_SECS} s"
    assert responses, "the retracted request raised; its traceback is in the worker log"
    out = _completed(responses[0])
    stats = out["meta_info"]["routed_experts_forcing"]
    # With routed_experts_start_len 0 and one retract followed by a cache-flushing update, forced_tokens is exactly
    # the snapshot: the rows the request computed under w0 before the retract. A second retract would force them
    # again and count them twice, so the bound below holds for this single-retract case only.
    assert stats["retract_snapshots"] == 1, f"want one retract snapshot mid-decode: {stats}"
    assert stats["unforced_tokens"] == 0 and stats["forced_tokens"] > len(prompt_ids), stats
    snapshot = stats["forced_tokens"]
    exact = _leading_equal_rows(*references)
    assert exact >= snapshot, (
        f"inconclusive: two uninterrupted w0 runs part at row {exact}, inside the {snapshot}-row snapshot, so neither "
        "is an exact reference; lower _RETRACT_DELAY_SECS"
    )
    agree = _leading_equal_rows(_routes(out, shape), references[0])
    assert agree >= snapshot, f"rows computed before the retract changed: {agree} < {snapshot}: {stats}"
    return {"forced": stats, "reference_exact_rows": exact, "leading_equal_rows": agree}


_ABSENT = "<absent>"
# run_ab.py's --rollout-max-response-len and sampling flags, through agentic_tool_call's own mapping.
_TURN_TOKENS = 2048
_ROLLOUT_SAMPLING = dict(
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    max_new_tokens=_TURN_TOKENS,
    stop=None,
    stop_token_ids=None,
    skip_special_tokens=False,
    no_stop_trim=True,
    spaces_between_special_tokens=False,
)
# What agentic_tool_call's tracer posts to create a training session (CreateSessionRequest minus the unset fields).
_SESSION_DEFAULTS = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
# httpx's default request headers, which agent.py's client sends and the session server forwards.
_AGENT_HEADERS = {
    "accept": "*/*",
    "accept-encoding": "gzip, deflate",
    "connection": "keep-alive",
    "user-agent": "python-httpx/0.28.1",
    "content-type": "application/json",
}
_CHAT_PROBLEM = "Let $x$ be the smallest positive integer such that $x^2 + 7x + 12$ is divisible by 18. Find $x$."
# sglang_router.launch_router as run_ab.py's router starts it (its --policy, --assignment-mode and health settings).
_SGLANG_ROUTER_ARGS = (
    "--policy",
    "manual",
    "--assignment-mode",
    "min_load",
    "--log-level",
    "warn",
    "--request-timeout-secs",
    "14400",
    "--health-failure-threshold",
    "40",
    "--health-success-threshold",
    "1",
    "--health-check-interval-secs",
    "15",
)
_SGLANG_ROUTER_READY_SECS = 300
_HOP_HEADERS = frozenset(
    ("host", "content-length", "transfer-encoding", "connection", "keep-alive", "content-encoding", "accept-encoding")
)


def _check_chat_session(
    url: str, shape: tuple[int, int], model: Path, tito_model: str, run_dir: Path, layout: str, *, base_port: int
) -> dict:
    """Every turn runs and is recorded; the result passes only if every turn carried honored forcing counters."""
    raw: list[dict] = []
    routes = {"direct": asyncio.run(_chat_route(url, shape, model, tito_model, capture=None, raw=raw))}
    turn1_ids = routes["direct"].pop("turn1_input_ids", None)
    if turn1_ids:
        routes["direct"]["cases"]["generate_control"] = _generate_control(url, turn1_ids)
    capture = router = None
    try:
        capture = _CaptureProxy(url, base_port=base_port)
        router = _SGLangRouter(capture.url, run_dir / f"router-{layout}.log", base_port=base_port)
        routes["router"] = asyncio.run(_chat_route(router.url, shape, model, tito_model, capture=capture, raw=raw))
        routes["router"].pop("turn1_input_ids", None)
    except Exception as error:
        routes["router"] = {"error": f"{type(error).__name__}: {error}", "cases": {}}
    finally:
        if router is not None:
            router.stop()
        if capture is not None:
            capture.close()
    (run_dir / f"chat-session-{layout}.json").write_text(json.dumps(raw, indent=1) + "\n")

    summary, failures = {}, []
    for route, outcome in routes.items():
        if "error" in outcome:
            failures.append(f"{route}: {outcome['error']}")
        for case, record in outcome["cases"].items():
            forcing = record.get("routed_experts_forcing", _ABSENT)
            summary[f"{route}/{case}"] = {
                "forcing": forcing,
                "finish_reason": record.get("finish_reason"),
                "session_server_verdict": record.get("session_server_verdict"),
            }
            honored = isinstance(forcing, dict) and forcing.get("unforced_tokens") == 0
            if case != "generate_control":
                honored = honored and record.get("session_server_verdict") == "honored"
            if not honored:
                failures.append(f"{route}/{case}")
    return {"passed": not failures, "failures": failures, "summary": summary, "routes": routes}


async def _chat_route(url: str, shape: tuple[int, int], model: Path, tito_model: str, *, capture, raw: list) -> dict:
    """Three turns through Miles' SessionCore and transport, with ``url`` as the session server's backend: (1) a first
    turn that stops early, (2) a first turn capped by max_tokens, (3) a second turn on (2).
    """
    import aiohttp
    import orjson
    from examples.experimental.routing_replay_forcing_ab.agent import NUDGE, SYSTEM_PROMPT

    import miles.rollout.session.core as session_core
    from miles.rollout.generate_hub.agentic_tool_call import build_chat_request_kwargs
    from miles.rollout.session.errors import RoutingReplayForcingRefusedError

    server = _session_server(url, shape, model, tito_model)
    core = server._core
    exchanges, verdicts = [], []
    transport, honored_check = server.do_proxy, session_core._check_routed_experts_forcing_honored

    async def recording_transport(request, path, *, body, headers):
        result = await transport(request, path, body=body, headers=headers)
        exchanges.append({**result, "sent_headers": headers})
        return result

    def recording_check(meta_info: dict) -> None:
        try:
            honored_check(meta_info)
        except RoutingReplayForcingRefusedError as error:
            verdicts.append(f"fail closed: {error}")
        else:
            verdicts.append("honored")

    async def new_session() -> str:
        return json.loads((await core.create_session(sampling_defaults=dict(_SESSION_DEFAULTS))).body)["session_id"]

    async def turn(session_id: str, case: str, client_body: dict) -> tuple[dict, dict | None, dict, dict | None]:
        """The turn's evidence, the client's reply, and the body the session server sent and the reply it got."""
        exchanges.clear()
        verdicts.clear()
        if capture is not None:
            capture.captured.clear()
        record, reply = {"session_id": session_id}, None
        try:
            response = await core.chat_completions(
                session_id, method="POST", query="", headers=dict(_AGENT_HEADERS), body=orjson.dumps(client_body)
            )
            record["client_status"] = response.status_code
            if response.status_code == 200:
                reply = json.loads(response.body)
            else:
                record["client_body"] = bytes(response.body).decode(errors="replace")[:2000]
        except Exception as error:
            record["client_status"] = getattr(error, "status_code", None)
            record["client_error"] = f"{type(error).__name__}: {error}"
        # None: the session server never ran its check (no key on the turn, or it failed before the check)
        record["session_server_verdict"] = verdicts[0] if verdicts else None
        request, upstream = {}, None
        if exchanges:
            exchange = exchanges[-1]
            request = orjson.loads(exchange["request_body"])
            record["sent_headers"] = {k: v for k, v in exchange["sent_headers"].items() if k.lower().startswith("x-")}
            record["request"] = _request_summary(request)
            record["status_code"] = exchange["status_code"]
            try:
                upstream = orjson.loads(exchange["response_body"])
                record.update(_reply_evidence(upstream, shape))
            except orjson.JSONDecodeError:
                record["response_text"] = bytes(exchange["response_body"]).decode(errors="replace")[:2000]
            raw_entry = {"case": case, "url": url, "request": request, "response": _trimmed(upstream)}
            if capture is not None:
                record["engine_side"] = [
                    _engine_side_evidence(request, sent, received, shape) for sent, received in capture.captured
                ]
                raw_entry["engine_side"] = [
                    {"request": _json_or_text(sent), "response": _trimmed(_json_or_text(received))}
                    for sent, received in capture.captured
                ]
            raw.append(raw_entry)
        return record, reply, request, upstream

    server.do_proxy = recording_transport
    session_core._check_routed_experts_forcing_honored = recording_check
    cases: dict[str, dict] = {}
    result = {"backend": url, "cases": cases}
    try:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=1024, keepalive_timeout=5), timeout=server.timeout
        ) as client:
            server.client = client
            base = build_chat_request_kwargs(dict(_ROLLOUT_SAMPLING))
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": _CHAT_PROBLEM}]
            # agent.py sets ignore_eos on 4-layer checkpoints, so every reply of a 4-layer run hits max_tokens.
            capped = {**base, "ignore_eos": True}
            session_a = await new_session()
            cases["2_turn1_length"], reply, request, upstream = await turn(
                session_a, "2_turn1_length", {**capped, "messages": messages}
            )
            result["turn1_input_ids"] = request.get("input_ids")
            if reply is not None:
                follow_up = [*messages, reply["choices"][0]["message"], {"role": "user", "content": NUDGE}]
                cases["3_turn2"], *_ = await turn(session_a, "3_turn2", {**capped, "messages": follow_up})
            else:
                cases["3_turn2"] = {"skipped": "turn (2) returned no reply to build on"}
            # Stop on any token turn (2) sampled: at temperature 1 the reply soon samples one again and stops early.
            stop_ids = sorted(_output_token_ids(upstream)) if upstream else []
            session_b = await new_session()
            cases["1_turn1_stop"], *_ = await turn(
                session_b, "1_turn1_stop", {**base, "stop_token_ids": stop_ids, "messages": messages}
            )
    finally:
        session_core._check_routed_experts_forcing_honored = honored_check
        await core.close()
    return result


def _session_server(url: str, shape: tuple[int, int], model: Path, tito_model: str):
    """A session server configured as run_ab.py's run configures it, but for its address and backend."""
    from miles.rollout.session.config import SessionServerConfig
    from miles.rollout.session.server import SessionServer
    from miles.utils.chat_template_utils import resolve_fixed_chat_template

    template_path, template_kwargs = resolve_fixed_chat_template(tito_model)
    config = SessionServerConfig(
        host="127.0.0.1",
        port=0,
        instance_id=None,
        backend_url=url,
        timeout=None,
        hf_checkpoint=str(model),
        chat_template_path=template_path,
        tito_model=tito_model,
        apply_chat_template_kwargs=template_kwargs,
        use_rollout_routing_replay=True,
        use_rollout_routing_replay_forcing=True,
        use_rollout_indexer_replay=False,
        use_sampling_support_replay=False,
        sglang_speculative_algorithm=None,
        num_layers=shape[0],
        moe_router_topk=shape[1],
        save_debug_trajectory_data=None,
        lora_rank=0,
        lora_adapter_path=None,
        lora_train_only=False,
        use_session_server=True,
        session_message_matcher="strict",
        pause_generation_mode="retract",
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
        rollout_top_logprobs_num=0,
        rollout_sampling_logprobs_mode="selected",
        rollout_temperature=1.0,
    )
    return SessionServer(config)


def _generate_control(url: str, input_ids: list[int]) -> dict:
    """A keyed /generate first turn on turn (2)'s prompt: the path the engine-smoke checks already exercise."""
    response = _generate(url, input_ids, _TURN_TOKENS, session_key=f"forcing-smoke-chat-control-{uuid.uuid4().hex}")
    record = {"status_code": response.status_code}
    if response.status_code != 200:
        return {**record, "response_text": response.text[:2000]}
    meta = response.json()["meta_info"]
    return {
        **record,
        "finish_reason": meta.get("finish_reason"),
        "meta_info_keys": sorted(meta),
        "routed_experts_forcing": meta.get("routed_experts_forcing", _ABSENT),
        "completion_tokens": meta.get("completion_tokens"),
    }


def _request_summary(request: dict) -> dict:
    return {
        key: f"<{len(value)} items>" if key in ("input_ids", "messages", "stop_token_ids") else value
        for key, value in sorted(request.items())
    }


def _reply_evidence(reply: dict, shape: tuple[int, int]) -> dict:
    choices = reply.get("choices") or [{}]
    choice = choices[0]
    meta = choice.get("meta_info")
    evidence = {
        "reply_keys": sorted(reply),
        "choice_keys": sorted(choice),
        "finish_reason": choice.get("finish_reason"),
        "meta_info_keys": sorted(meta) if isinstance(meta, dict) else None,
    }
    if isinstance(meta, dict):
        evidence["routed_experts_forcing"] = meta.get("routed_experts_forcing", _ABSENT)
        for key in ("finish_reason", "prompt_tokens", "completion_tokens", "cached_tokens", "num_retractions"):
            evidence[f"meta_{key}"] = meta.get(key)
        if isinstance(meta.get("routed_experts"), str):
            row_bytes = 4 * shape[0] * shape[1]
            evidence["routed_experts_rows"] = len(base64.b64decode(meta["routed_experts"])) // row_bytes
    return evidence


def _engine_side_evidence(sent_by_session: dict, sent: bytes, received: bytes, shape: tuple[int, int]) -> dict:
    """What the engine received from the router, against what the session server sent the router."""
    request, reply = _json_or_text(sent), _json_or_text(received)
    evidence = {}
    if isinstance(request, dict):
        evidence["request"] = _request_summary(request)
        evidence["dropped_by_router"] = sorted(set(sent_by_session) - set(request))
        evidence["added_by_router"] = sorted(set(request) - set(sent_by_session))
        evidence["changed_by_router"] = sorted(
            key for key in set(sent_by_session) & set(request) if sent_by_session[key] != request[key]
        )
    else:
        evidence["request_text"] = str(request)[:2000]
    if isinstance(reply, dict):
        evidence.update(_reply_evidence(reply, shape))
    else:
        evidence["response_text"] = str(reply)[:2000]
    return evidence


def _json_or_text(body: bytes):
    try:
        return json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return bytes(body).decode(errors="replace")


def _trimmed(reply):
    """The reply with its base64 routed-expert rows replaced by their length."""
    if not isinstance(reply, dict):
        return reply
    reply = json.loads(json.dumps(reply))
    for choice in reply.get("choices") or []:
        meta = choice.get("meta_info")
        if isinstance(meta, dict) and isinstance(meta.get("routed_experts"), str):
            meta["routed_experts"] = f"<{len(meta['routed_experts'])} base64 chars>"
    return reply


def _output_token_ids(reply: dict) -> set[int]:
    meta = (reply.get("choices") or [{}])[0].get("meta_info") or {}
    return {entry[1] for entry in meta.get("output_token_logprobs") or []}


class _CaptureProxy:
    """Forwards every request to the engine and keeps each chat completion body pair the router exchanged."""

    def __init__(self, upstream: str, *, base_port: int):
        self.captured: list[tuple[bytes, bytes]] = []
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:
                pass

            def _forward(self) -> None:
                body = self._read_body()
                headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_HEADERS}
                reply = requests.request(
                    self.command, f"{upstream}{self.path}", data=body, headers=headers, timeout=_HTTP_TIMEOUT_SECS
                )
                content = reply.content
                if self.path.startswith("/v1/chat/completions"):
                    proxy.captured.append((body, content))
                self.send_response(reply.status_code)
                for key, value in reply.headers.items():
                    if key.lower() not in _HOP_HEADERS:
                        self.send_header(key, value)
                self.send_header("content-length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def _read_body(self) -> bytes:
                if "chunked" in self.headers.get("transfer-encoding", "").lower():
                    chunks = []
                    while size := int(self.rfile.readline().split(b";")[0].strip(), 16):
                        chunks.append(self.rfile.read(size))
                        self.rfile.readline()
                    self.rfile.readline()
                    return b"".join(chunks)
                length = int(self.headers.get("content-length") or 0)
                return self.rfile.read(length) if length else b""

            do_GET = do_POST = do_PUT = do_DELETE = _forward

        port = find_available_port(base_port + 2000)
        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class _SGLangRouter:
    """The SGLang router with one worker, ready once a chat completion through it succeeds."""

    def __init__(self, worker_url: str, log_path: Path, *, base_port: int):
        port = find_available_port(base_port)
        prometheus_port = find_available_port(base_port + 1000)
        self.url = f"http://127.0.0.1:{port}"
        self._log_path = log_path
        self._log = log_path.open("w")
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang_router.launch_router",
                *_SGLANG_ROUTER_ARGS,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--prometheus-port",
                str(prometheus_port),
            ],
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
        try:
            self._wait(lambda: requests.get(f"{self.url}/liveness", timeout=5).status_code == 200, "liveness")
            requests.post(
                f"{self.url}/workers", json={"url": worker_url, "worker_type": "regular"}, timeout=60
            ).raise_for_status()
            probe = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
            self._wait(
                lambda: requests.post(
                    f"{self.url}/v1/chat/completions",
                    json=probe,
                    headers={"X-SMG-Routing-Key": "forcing-smoke-probe"},
                    timeout=60,
                ).status_code
                == 200,
                "a chat completion through the router",
            )
        except BaseException:
            self.stop()
            raise

    def _wait(self, ready, what: str) -> None:
        deadline = time.monotonic() + _SGLANG_ROUTER_READY_SECS
        last = None
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"the router exited with {self._process.returncode}; see {self._log_path.name}")
            try:
                if ready():
                    return
            except requests.RequestException as error:
                last = error
            time.sleep(2)
        raise TimeoutError(f"no {what} within {_SGLANG_ROUTER_READY_SECS} s (last error: {last})")

    def stop(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._log.close()


def _perturbed_checkpoint(model: Path, out: Path) -> Path:
    """Symlinked copy of the checkpoint with every MoE router weight scaled by 1 + _PERTURB_SCALE * N(0, 1)."""
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    generator = torch.Generator().manual_seed(0)
    changed = 0
    for source in sorted(model.iterdir()):
        routers = []
        if source.suffix == ".safetensors":
            with safe_open(source, framework="pt") as handle:
                routers = [key for key in handle.keys() if key.endswith(_ROUTER_SUFFIX)]
        if not routers:
            (out / source.name).symlink_to(source)
            continue
        tensors = load_file(source)
        for key in routers:
            noise = torch.randn(tensors[key].shape, generator=generator)
            tensors[key] = (tensors[key].float() * (1 + _PERTURB_SCALE * noise)).to(tensors[key].dtype)
        save_file(tensors, out / source.name, metadata={"format": "pt"})
        changed += len(routers)
    assert changed, f"no *{_ROUTER_SUFFIX} tensors in {model}"
    return out


def _post(url: str, path: str, body: dict) -> dict:
    response = requests.post(f"{url}/{path}", json=body, timeout=_HTTP_TIMEOUT_SECS)
    response.raise_for_status()
    return response.json()


def _update_weights(url: str, model_path: Path) -> None:
    result = _post(url, "update_weights_from_disk", {"model_path": str(model_path)})
    assert result["success"], result


def _tokenize(url: str, text: str, *, add_special_tokens: bool) -> list[int]:
    return _post(url, "tokenize", {"prompt": text, "add_special_tokens": add_special_tokens})["tokens"]


def _long_prompt_ids(url: str) -> list[int]:
    lines = (
        _FILLER_LINE.format(i=i, a=(i * 7919) % 10007, b=(i * 104729) % 1009, c=(i * 31) % 97)
        for i in range(_CONTEXT_TOKENS // 8)
    )
    filler = _tokenize(url, "\n".join(lines), add_special_tokens=True)[:_CONTEXT_TOKENS]
    return filler + _tokenize(url, "\n\n" + _PROMPT, add_special_tokens=False)


def _generate(
    url: str,
    input_ids: list[int],
    max_new_tokens: int,
    *,
    session_key: str | None = None,
    start_len: int = 0,
    extra_key: str | None = None,
) -> requests.Response:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens, "ignore_eos": True},
        "return_logprob": True,
        "return_routed_experts": True,
        "routed_experts_start_len": start_len,
    }
    if session_key is not None:
        payload["routed_experts_session_key"] = session_key
    if extra_key is not None:
        payload["extra_key"] = extra_key
    return requests.post(f"{url}/generate", json=payload, timeout=_HTTP_TIMEOUT_SECS)


def _completed(response: requests.Response) -> dict:
    response.raise_for_status()
    output = response.json()
    finish_reason = output["meta_info"]["finish_reason"]
    assert finish_reason["type"] != "abort", finish_reason
    return output


def _refusal(response: requests.Response) -> dict:
    """A refusal before compute: HTTP 200, an abort finish, no token, and the reason code ahead of the ":"."""
    response.raise_for_status()
    meta = response.json()["meta_info"]
    assert meta["finish_reason"]["type"] == "abort", f"an unheld session key was admitted: {meta['finish_reason']}"
    assert meta["completion_tokens"] == 0, meta
    return {"code": meta["routed_experts_forcing"]["refused"], "message": meta["finish_reason"]["message"]}


def _output_ids(output: dict) -> list[int]:
    return [entry[1] for entry in output["meta_info"]["output_token_logprobs"]]


def _routes(output: dict, shape: tuple[int, int]) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(output["meta_info"]["routed_experts"]), dtype=np.int32)
    return raw.reshape(-1, *shape)


def _differing_row_frac(a: np.ndarray, b: np.ndarray) -> float:
    """Share of (row, layer) expert selections whose set differs."""
    return float((np.sort(a, axis=-1) != np.sort(b, axis=-1)).any(axis=-1).mean())


def _leading_equal_rows(a: np.ndarray, b: np.ndarray) -> int:
    """How many leading rows select the same set of experts in every layer."""
    rows = min(len(a), len(b))
    agree = (np.sort(a[:rows], axis=-1) == np.sort(b[:rows], axis=-1)).all(axis=(1, 2))
    return rows if agree.all() else int(np.argmin(agree))


def _summary(evidence: dict) -> bool:
    for result in evidence["layouts"]:
        print(f"[{'PASS' if result['passed'] else 'FAIL'}] {result['layout']}: {result.get('error', '')}", flush=True)
    return all(result["passed"] for result in evidence["layouts"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", required=True, help="HF checkpoint with MoE routers (*.mlp.gate.weight)")
    parser.add_argument("--tito-model", required=True, help="the session server's --tito-model for this checkpoint")
    parser.add_argument(
        "--engine-args",
        required=True,
        help="sglang.launch_server flags of the run as one string, spelled --engine-args='...'; the smoke adds the "
        "forcing and radix-cache flags",
    )
    parser.add_argument("--out-dir", default="engine-smoke-out")
    parser.add_argument(
        "--base-port",
        type=int,
        default=_DEFAULT_BASE_PORT,
        help="first port: engine from here, router +2000, its metrics +3000, capture proxy +4000",
    )
    parser.add_argument("--work-dir", help="where the router-perturbed checkpoint copy goes; default: a new temp dir")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="forcing-smoke-") as scratch:
        evidence = run(
            model_path=args.model_path,
            engine_args=shlex.split(args.engine_args),
            out_dir=args.out_dir,
            tito_model=args.tito_model,
            base_port=args.base_port,
            work_dir=args.work_dir or scratch,
        )
    return 0 if _summary(evidence) else 1


def test_engine_smoke():
    import pytest

    model_path = os.environ.get("ENGINE_SMOKE_MODEL_PATH")
    if not model_path:
        pytest.skip("set ENGINE_SMOKE_MODEL_PATH, ENGINE_SMOKE_TITO_MODEL and ENGINE_SMOKE_ENGINE_ARGS")
    with tempfile.TemporaryDirectory(prefix="forcing-smoke-") as work_dir:
        evidence = run(
            model_path=model_path,
            engine_args=shlex.split(os.environ["ENGINE_SMOKE_ENGINE_ARGS"]),
            out_dir=os.environ.get("ENGINE_SMOKE_OUT_DIR", "engine-smoke-out"),
            tito_model=os.environ["ENGINE_SMOKE_TITO_MODEL"],
            base_port=int(os.environ.get("ENGINE_SMOKE_BASE_PORT", _DEFAULT_BASE_PORT)),
            work_dir=work_dir,
        )
    assert _summary(evidence), json.dumps(evidence["layouts"], indent=1)[:4000]


if __name__ == "__main__":
    sys.exit(main())
