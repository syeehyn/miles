"""Multi-turn Python-tool agent for the routing replay forcing A/B (run_ab.py). Test-only.

One episode is one session-server conversation about a dapo-math problem. The model may end a reply with a Python
code block; the next user turn shows what it printed. Until the episode has ROUTING_REPLAY_AB_MIN_TURNS turns, a reply
without code gets a fixed nudge instead of ending the episode, so every session is re-entered several times. With
ROUTING_REPLAY_AB_IGNORE_EOS=1 (run_ab.py sets it for 4-layer checkpoints) every reply runs to the turn cap. The tool
protocol is plain text, so the agent works with any chat template. The reward is examples/retool_v2's reward_func,
which adds a tool-use bonus from the tool_call_count returned here.
"""

import asyncio
import os
import re
import sys
import tempfile
from pathlib import Path

import httpx
from examples.retool_v2.tool_sandbox import PythonSandbox

from miles.rollout.agentic import rollout_limits

MAX_TURNS = 8
NUDGE = "Check your work once more, then continue."
SYSTEM_PROMPT = (
    "You can run Python to check your work. To run code, end your reply with one ```python block; the next message "
    "shows what it printed. When you are sure, give the final answer as \\boxed{answer} and no code."
)
# Set by run_ab.py.
_MIN_TURNS_ENV = "ROUTING_REPLAY_AB_MIN_TURNS"
_IGNORE_EOS_ENV = "ROUTING_REPLAY_AB_IGNORE_EOS"
_TOOL_OUTPUT_CHARS = 4000
_HTTP_TIMEOUT_SECS = 1800
_CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
# Model-written code runs in an isolated interpreter that sees a scratch directory and no environment: the rollout's
# environment can hold credentials. The kernel caps its memory and output file, and a timeout its wall time.
_PYTHON_SLOTS = 32
_PYTHON_TIMEOUT_SECS = 60
_PYTHON_MEMORY_BYTES = 1024**3
_PYTHON_OUTPUT_BYTES = 1024**2
_PYTHON_PRELUDE = (
    "import resource as _r; "
    f"_r.setrlimit(_r.RLIMIT_AS, ({_PYTHON_MEMORY_BYTES}, {_PYTHON_MEMORY_BYTES})); "
    f"_r.setrlimit(_r.RLIMIT_FSIZE, ({_PYTHON_OUTPUT_BYTES}, {_PYTHON_OUTPUT_BYTES})); "
    "del _r\n"
)
# examples/retool_v2's sandbox filter: the tool refuses the code the retool example refuses
_SANDBOX = PythonSandbox()


async def run_agent(base_url: str, prompt: list[dict], request_kwargs: dict, metadata: dict, **kwargs) -> dict:
    min_turns = int(os.environ.get(_MIN_TURNS_ENV, "0"))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *prompt]
    # --max-seq-len is the session's token budget, --rollout-max-response-len one reply's cap.
    token_budget = metadata["max_seq_len"]
    turn_tokens = request_kwargs["max_tokens"]
    request = {**request_kwargs, "ignore_eos": True} if os.environ.get(_IGNORE_EOS_ENV) == "1" else request_kwargs
    tool_calls = turns = 0
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECS) as client:
        while True:
            response = await client.post(f"{base_url}/v1/chat/completions", json={**request, "messages": messages})
            response.raise_for_status()
            body = response.json()
            message = body["choices"][0]["message"]
            # The next request must extend the recorded conversation exactly, so the reply goes back as returned.
            messages.append(message)
            turns += 1
            # A next user turn plus a full reply must still fit the session's budget.
            if turns == MAX_TURNS or body["usage"]["total_tokens"] + 2 * turn_tokens > token_budget:
                break
            code = _code_to_run(message.get("content") or "")
            if code is not None:
                tool_calls += 1
                # The sandbox answers "Output:\n..." or "Error: ..."; the tail keeps the context bounded.
                messages.append({"role": "user", "content": (await _run_python(code))[-_TOOL_OUTPUT_CHARS:]})
            elif turns < min_turns:
                messages.append({"role": "user", "content": NUDGE})
            else:
                break
    return {"tool_call_count": tool_calls, "agent_turns": turns}


def _code_to_run(content: str) -> str | None:
    """The reply's last code block, unless the reply already gives its final answer."""
    if "\\boxed" in content:
        return None
    blocks = _CODE_BLOCK.findall(content)
    return blocks[-1] if blocks else None


async def _run_python(code: str) -> str:
    safe, reason = _SANDBOX._check_code_safety(code)
    if not safe:
        return f"Error: {reason}"
    # rollout_limits caps the interpreters of every episode of this rollout process at once
    async with rollout_limits.rollout_semaphore("routing-replay-ab-python", _PYTHON_SLOTS):
        with tempfile.TemporaryDirectory(prefix="routing_replay_ab_") as scratch:
            script, output = Path(scratch) / "code.py", Path(scratch) / "output.txt"
            script.write_text(_PYTHON_PRELUDE + code)
            with output.open("wb") as sink:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-I", str(script), cwd=scratch, env={}, stdout=sink, stderr=sink
                )
                try:
                    returncode = await asyncio.wait_for(process.wait(), _PYTHON_TIMEOUT_SECS)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                    return f"Error: Code execution timed out after {_PYTHON_TIMEOUT_SECS} seconds"
            printed = output.read_text(errors="replace").strip()
    return f"Output:\n{printed}" if returncode == 0 else f"Error: Process exited with code {returncode}\n{printed}"
