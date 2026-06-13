"""The brain of the demo: a Google ADK agent that can USE TOOLS.

The big idea
------------
A language model (Gemini) is great at *talking*, but on its own it can't do
real things: it can't reliably do math, it doesn't know today's date, and it
can't look up live weather. So we give it "tools" — ordinary Python functions.

Gemini reads each function's name, its arguments, and its docstring, and
decides on its own WHEN to call which tool. That is what turns a chatbot into
an *agent*.

How we reach Gemini
-------------------
We use a FREE Google AI Studio API key (get one at
https://aistudio.google.com/apikey). No service account, no billing. The
single GOOGLE_API_KEY loaded in load_env() below is all the ADK needs to
talk to Gemini.
"""
from __future__ import annotations

import ast
import asyncio
import operator
import os
import threading
from datetime import datetime

import requests
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

MODEL = "gemini-2.5-flash"  # fast, cheap, free-tier friendly
APP_NAME = "tool_demo"

# When True, every step (user message, tool calls + arguments, tool results,
# final answer) is printed to the terminal. Great for showing students what
# the agent is "thinking". Set TRACE=0 in the environment to silence it.
TRACE = os.environ.get("TRACE", "1") != "0"


def _trace(line: str = "") -> None:
    if TRACE:
        print(line, flush=True)


# Google's servers occasionally return 503 "high demand" / 429 rate-limit
# errors. These are temporary, so we retry a few times before giving up.
MAX_ATTEMPTS = 3
_TRANSIENT_SIGNS = ("unavailable", "overloaded", "high demand", "try again",
                    "deadline", "rate limit", "resource exhausted")


def _is_transient(exc: BaseException) -> bool:
    """True if the error is likely temporary and worth retrying."""
    code = getattr(exc, "code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    blob = (repr(exc) + " " + str(exc)).lower()
    return any(sign in blob for sign in _TRANSIENT_SIGNS)


def _friendly_error(exc: BaseException) -> str:
    """A short, human-readable message for the chat bubble."""
    if _is_transient(exc):
        return ("⚠️ Gemini is busy right now (Google's servers are under high "
                "demand). I tried a few times — please send your message again "
                "in a few seconds.")
    return f"⚠️ Something went wrong: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# 1. THE TOOLS  (just normal Python functions)
# ---------------------------------------------------------------------------
# Rules ADK cares about:
#   * Use clear names and type hints  -> Gemini uses these to fill arguments.
#   * Write a good docstring          -> Gemini reads it to know what the tool
#                                         does and when to use it.
#   * Return a dict or a string       -> this goes back to Gemini as the result.

# Operators we allow in the calculator. We do NOT use Python's eval() because
# eval() can run ANY code (dangerous). This whitelist only allows arithmetic.
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Only numbers and + - * / ** % are allowed.")


def calculator(expression: str) -> dict:
    """Do exact arithmetic. Use this for ANY math instead of guessing.

    Args:
        expression: A math expression, e.g. "12 * (7 + 3)" or "2 ** 10".

    Returns:
        A dict with the numeric result, or an error message.
    """
    try:
        result = _safe_eval(ast.parse(expression, mode="eval").body)
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"expression": expression, "error": str(exc)}


def get_current_datetime() -> dict:
    """Return the current date, time and weekday on the server.

    Use this whenever the user asks about the current time, today's date,
    or what day of the week it is. The model has no built-in clock.
    """
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
    }


def count_letters(word: str, letter: str) -> dict:
    """Count how many times a letter appears in a word (case-insensitive).

    Use this for 'how many X's in Y' questions. Language models are bad at
    counting letters by themselves, so this tool gives the exact answer.

    Args:
        word: The word to inspect, e.g. "strawberry".
        letter: The single letter to count, e.g. "r".
    """
    count = word.lower().count(letter.lower())
    return {"word": word, "letter": letter, "count": count}


# Map Open-Meteo weather codes to plain English (just the common ones).
_WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "rime fog", 51: "light drizzle", 61: "light rain",
    63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
    75: "heavy snow", 80: "rain showers", 95: "thunderstorm",
}


def get_weather(city: str) -> dict:
    """Look up the LIVE current weather for a city using a free public API.

    This calls the internet in real time (Open-Meteo, no API key needed),
    so the answer is real data, not something the model imagined.

    Args:
        city: The city name, e.g. "London" or "Mumbai".
    """
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()
        if not geo.get("results"):
            return {"city": city, "error": "City not found."}
        place = geo["results"][0]
        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,weather_code,wind_speed_10m",
            },
            timeout=10,
        ).json()["current"]
        return {
            "city": f"{place['name']}, {place.get('country', '')}".strip(", "),
            "temperature_celsius": weather["temperature_2m"],
            "wind_speed_kmh": weather["wind_speed_10m"],
            "conditions": _WEATHER_CODES.get(weather["weather_code"], "unknown"),
        }
    except Exception as exc:
        return {"city": city, "error": f"Could not fetch weather: {exc}"}


TOOLS = [calculator, get_current_datetime, count_letters, get_weather]


# ---------------------------------------------------------------------------
# 2. THE AGENT  (Gemini model + instructions + the tools)
# ---------------------------------------------------------------------------
def load_env() -> None:
    """Load the free Gemini API key from the .env file into the environment."""
    # Read .env (a tiny key=value file) if present, so students don't have to
    # export variables by hand. ADK picks up GOOGLE_API_KEY automatically.
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


def build_agent() -> Agent:
    return Agent(
        name="helpful_tool_agent",
        model=MODEL,  # a plain string -> ADK uses the free Gemini endpoint
        description="A friendly assistant that can use tools to get real answers.",
        instruction=(
            "You are a friendly, concise assistant for students. "
            "You have tools for math, the current date/time, counting letters "
            "in a word, and live weather. ALWAYS use the relevant tool instead "
            "of guessing — especially for math, dates, letter-counting and "
            "weather. After using a tool, explain the answer in one short, "
            "clear sentence."
        ),
        tools=TOOLS,
    )


# ---------------------------------------------------------------------------
# 3. RUNNING THE AGENT
# ---------------------------------------------------------------------------
# ADK is async. A web server handles many requests, so we keep ONE background
# event loop alive for the whole app and send every agent call to it. (This
# avoids "event loop is closed" errors that happen if you spin up a new loop
# per request.)
load_env()
_agent = build_agent()
_session_service = InMemorySessionService()
_runner = Runner(agent=_agent, app_name=APP_NAME, session_service=_session_service)

_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True).start()
_known_sessions: set[str] = set()


def _run_coro(coro):
    """Run an async coroutine on the shared background loop and wait for it."""
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


async def _run_once(session_id: str, user_message: types.Content) -> dict:
    """One full agent turn: stream events, trace them, collect the reply."""
    reply_text = ""
    tools_used: list[dict] = []  # we collect these so the UI can SHOW them
    async for event in _runner.run_async(
        user_id="student", session_id=session_id, new_message=user_message
    ):
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            # The model decided to call a tool:
            if getattr(part, "function_call", None):
                fc = part.function_call
                args = dict(fc.args or {})
                tools_used.append({"name": fc.name, "args": args})
                arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                _trace(f"  🔧  TOOL CALL → {fc.name}({arg_str})")
            # The tool's result coming back to the model:
            if getattr(part, "function_response", None):
                fr = part.function_response
                _trace(f"  📦  TOOL RESULT ← {fr.name}: {fr.response}")
            # The model's final words to the user:
            if getattr(part, "text", None):
                reply_text += part.text

    _trace(f"🤖  AGENT: {reply_text.strip()}")
    if tools_used:
        _trace(f"     (tools used: {', '.join(t['name'] for t in tools_used)})")
    return {"reply": reply_text.strip(), "tools_used": tools_used}


async def _ask_async(session_id: str, message: str) -> dict:
    # Create the conversation's memory the first time we see this session id.
    if session_id not in _known_sessions:
        await _session_service.create_session(
            app_name=APP_NAME, user_id="student", session_id=session_id
        )
        _known_sessions.add(session_id)

    user_message = types.Content(role="user", parts=[types.Part(text=message)])

    _trace("\n" + "═" * 64)
    _trace(f"👤  USER ({session_id}): {message}")

    # Retry transient Google errors (503 high-demand, 429 rate-limit) with a
    # short, growing back-off. Anything else fails fast.
    last_exc: BaseException | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = await _run_once(session_id, user_message)
            _trace("═" * 64 + "\n")
            return result
        except Exception as exc:  # noqa: BLE001 - we deliberately never crash
            last_exc = exc
            if _is_transient(exc) and attempt < MAX_ATTEMPTS:
                wait = 1.5 * attempt
                _trace(f"  ⚠️  {type(exc).__name__} (transient) — retry "
                       f"{attempt}/{MAX_ATTEMPTS - 1} in {wait:.1f}s")
                await asyncio.sleep(wait)
                continue
            break

    # Every attempt failed — return a clean JSON error, never a 500/crash.
    _trace(f"  ❌  GAVE UP: {type(last_exc).__name__}: {last_exc}")
    _trace("═" * 64 + "\n")
    return {"reply": _friendly_error(last_exc), "tools_used": [], "error": True}


def ask(session_id: str, message: str) -> dict:
    """Send a message to the agent and get back its reply + tools it used.

    This NEVER raises — on any failure it returns a friendly error dict, so the
    web layer can always reply with valid JSON instead of a 500 crash.
    """
    try:
        return _run_coro(_ask_async(session_id, message))
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        _trace(f"  ❌  UNEXPECTED: {type(exc).__name__}: {exc}")
        return {"reply": _friendly_error(exc), "tools_used": [], "error": True}


# Lets you test the agent straight from the terminal, no web server needed:
#   python agent.py
if __name__ == "__main__":
    print("Tool agent ready. Type 'quit' to exit.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue
        result = ask("terminal-session", text)
        if result["tools_used"]:
            names = ", ".join(t["name"] for t in result["tools_used"])
            print(f"  (used tools: {names})")
        print(f"Agent: {result['reply']}\n")
