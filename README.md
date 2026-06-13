# 🤖 AI Agent Demo — Google ADK + Gemini

A beginner-friendly demo for students. It shows the single most important idea
in modern AI: **giving a language model *tools* so it can do real things**, not
just chat.

You ask a question in a web page → Google's **Gemini** model decides whether it
needs a tool → it calls one of our Python functions (math, date, letter-counting,
or **live weather**) → and answers you. The page even shows you *which* tool the
agent used.

Built with **Google ADK** (Agent Development Kit) and a **free Gemini API key**
— no service account, no billing.

---

## 1. Get a free Gemini API key (30 seconds)

1. Go to **https://aistudio.google.com/apikey**
2. Sign in with any Google account and click **Create API key**.
3. Copy the key.

The free tier (~15 requests/minute) is plenty for a classroom.

## 2. Set up the project

```bash
# Create a virtual environment and install the libraries
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Add your key
cp .env.example .env
# then open .env and paste your key after GOOGLE_API_KEY=
```

## 3. Run it

**Web page (the main demo):**
```bash
python app.py
```
Open **http://localhost:5000** and start chatting.

**Or try it in the terminal (no web page):**
```bash
python agent.py
```

---

## Try asking

- `What's 23 × 47 + 100?`  → uses the **calculator** tool
- `How many r's in strawberry?`  → uses the **count_letters** tool
- `What's the weather in Tokyo?`  → uses the **live weather** tool (real internet data!)
- `What day is it today?`  → uses the **date/time** tool

Watch the little `🔧 used: ...` tag under each reply — that's the agent choosing
a tool on its own.

---

## How it works (the 3 pieces)

| File | What it is |
|------|------------|
| [`agent.py`](agent.py) | The **agent**: the Gemini model + its instructions + the tool functions. This is the heart of the demo. |
| [`app.py`](app.py) | A small **Flask web server** that connects the browser to the agent. |
| [`templates/index.html`](templates/index.html) | The **chat web page** the students see. |

The whole "agent" is just this:

```python
from google.adk.agents import Agent

agent = Agent(
    name="helpful_tool_agent",
    model="gemini-2.5-flash",        # the brain (free Gemini)
    instruction="You are a helpful assistant... always use tools.",
    tools=[calculator, get_current_datetime, count_letters, get_weather],
)
```

Gemini reads each tool's **name, arguments, and docstring** and decides when to
call it. That's the magic — *you* never write "if the user asks for math, call
the calculator." The model figures it out.

## Want to add your own tool?

1. Write a normal Python function in `agent.py` with a clear name, type hints,
   and a docstring.
2. Add it to the `TOOLS = [...]` list.
3. Restart. Done — the agent can now use it.
