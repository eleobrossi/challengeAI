#!/usr/bin/env python3
"""
test_langfuse.py

Minimal connectivity test — copy of the tutorial's "Run a Single Traced Call"
example. Run this LOCALLY (not via Cowork) to verify that your API keys work
and that traces appear on the Langfuse dashboard.

Usage (from your local terminal, inside the challenge folder):

    # 1. Create and activate a virtual environment
    python3 -m venv venv
    source venv/bin/activate          # Mac / Linux
    venv\\Scripts\\activate.bat       # Windows

    # 2. Install dependencies (same as requirements.txt)
    pip install langchain langchain-openai "langfuse==3.6.2" python-dotenv ulid-py

    # 3. Run this test
    python test_langfuse.py

If successful you will see the session ID printed, and within a few seconds
the trace will appear on https://challenges.reply.com/langfuse
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── same imports as tutorial ──────────────────────────────────────────────────
import ulid
# CRITICAL FIX: use get_client() not Langfuse() so that langfuse_client is the
# SAME singleton used internally by @observe().  A separate Langfuse() instance
# would cause update_current_trace() to miss the active trace.
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ── model setup (tutorial Step 1) ────────────────────────────────────────────
model_id = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")

model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    model=model_id,
    temperature=0.7,
    max_tokens=200,
)
print(f"✓ Model configured: {model_id}")

# ── Langfuse setup (tutorial Step 2) ─────────────────────────────────────────
# get_client() returns the global default singleton — same object @observe() uses
langfuse_client = get_client()


def generate_session_id():
    """Generate a unique session ID using TEAM_NAME and ULID."""
    return f"{os.getenv('TEAM_NAME', 'tutorial')}-{ulid.new().str}"


def invoke_langchain(model, prompt, langfuse_handler):
    """Invoke LangChain with the given prompt and Langfuse handler."""
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages, config={"callbacks": [langfuse_handler]})
    return response.content


@observe()
def run_llm_call(session_id, model, prompt):
    """Run a single LangChain invocation and track it in Langfuse."""
    # Update trace with session_id
    langfuse_client.update_current_trace(session_id=session_id)

    # Create Langfuse callback handler for automatic generation tracking
    langfuse_handler = CallbackHandler()

    # Invoke LangChain with Langfuse handler to track tokens and costs
    response = invoke_langchain(model, prompt, langfuse_handler)
    return response


print("✓ Langfuse initialized")
print(f"✓ Public key: {os.getenv('LANGFUSE_PUBLIC_KEY', 'NOT SET')[:20]}...")
print(f"✓ Host: {os.getenv('LANGFUSE_HOST', 'NOT SET')}")
print()

# ── single traced call (tutorial "Run a Single Traced Call") ─────────────────
session_id = generate_session_id()
print(f"Session ID: {session_id}\n")

response = run_llm_call(session_id, model, "What is the square root of 144?")

print(f"Input:    What is the square root of 144?")
print(f"Response: {response}")

langfuse_client.flush()

print()
print("✓ Trace sent to Langfuse with full token usage and cost data")
print(f"✓ Session ID: {session_id}")
print(f"✓ Check dashboard: {os.getenv('LANGFUSE_HOST', '')}")
print()

# ── multiple calls, same session (tutorial "Track Multiple Calls") ────────────
questions = [
    "What is machine learning?",
    "Explain neural networks in one sentence.",
    "What is the difference between AI and ML?",
]

session_id2 = generate_session_id()
print(f"Multi-call session ID: {session_id2}")
print(f"Making {len(questions)} calls...\n")

for i, question in enumerate(questions, 1):
    resp = run_llm_call(session_id2, model, question)
    print(f"Call {i}: {question}")
    print(f"  → {resp[:100]}...\n")

langfuse_client.flush()

print("=" * 60)
print(f"✓ All {len(questions)} traces sent!")
print(f"✓ Session: {session_id2}")
print(f"✓ Open dashboard → Sessions → look for your TEAM_NAME prefix")
print("=" * 60)
