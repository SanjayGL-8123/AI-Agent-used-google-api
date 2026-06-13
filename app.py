from flask import Flask, jsonify, render_template, request

import agent

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    # The browser sends a stable id so the agent remembers the conversation.
    session_id = data.get("session_id") or "web-session"
    if not message:
        return jsonify({"reply": "Please type something!", "tools_used": []})
    # agent.ask() is designed never to raise, but wrap it anyway so the browser
    # ALWAYS gets valid JSON (never an empty 500 page that breaks JSON.parse).
    try:
        result = agent.ask(session_id, message)
    except Exception as exc:  # noqa: BLE001
        result = {"reply": f"⚠️ Server error: {exc}", "tools_used": [], "error": True}
    return jsonify(result)


if __name__ == "__main__":
    # debug=False so the agent (and its background loop) isn't created twice.
    app.run(host="0.0.0.0", port=5000, debug=False)
