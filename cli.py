from my_ai import HybridMemorySystem
import os
import shutil
from openai import OpenAI
from dotenv import load_dotenv

import sqlite3
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime
import re
import hashlib
import shlex

# Load API key from .env file (create this file with: OPENAI_API_KEY=your_key_here)
load_dotenv()

# Lazily construct the OpenAI client so tests can run without a real API key.
_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        # When running in offline/test mode, allow the client to be omitted.
        if not api_key and not os.getenv("AI_MEMORY_OFFLINE"):
            raise RuntimeError("OPENAI_API_KEY must be set or AI_MEMORY_OFFLINE enabled")
        _client = OpenAI(api_key=api_key)
    return _client

def generate_response(query: str, context: str) -> str:
    """Call the LLM, with an optional offline mode for tests."""
    # Offline/test mode: skip real network calls when AI_MEMORY_OFFLINE is set.
    if os.getenv("AI_MEMORY_OFFLINE"):
        if context:
            return f"[offline-test] Using context of length {len(context)} for: {query[:50]}"
        else:
            return f"[offline-test] No context for: {query[:50]}"

    system_prompt = """
You are a helpful assistant with memory of past conversations.

When answering about the user (their pets, preferences, biography, etc.):
- Treat the section 'Relevant context about the user' as the ONLY source of truth.
- Do NOT invent or speculate about any user-specific details (e.g. personality traits, habits, activities)
  that are not explicitly stated in that context.
- If the user asks for a detail that is not present in the context, say you don't know or that it
  has not been mentioned yet.

You may still use general world knowledge for non-user-specific questions.
""".strip()

    if context:
        system_prompt += f"\n\nRelevant context about the user:\n{context}"

    try:
        backend = os.getenv("AI_BACKEND", "openai").lower()

        # Allow model and temperature to be configured via environment while keeping
        # sensible defaults for tests and local usage.
        if backend == "ollama":
            # Priority: explicit OLLAMA_MODEL, then AI_MODEL, then a sensible local default.
            model = os.getenv("OLLAMA_MODEL") or os.getenv("AI_MODEL") or "llama3.2:3b"
        else:
            model = os.getenv("AI_MODEL", "gpt-3.5-turbo")

        try:
            temperature = float(os.getenv("AI_TEMPERATURE", "0.2"))
        except ValueError:
            temperature = 0.2

        if backend == "ollama":
            import requests

            base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
            url = f"{base_url}/api/chat"

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                "stream": False,
            }

            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # Ollama's /api/chat returns a single message object when stream=False.
            return data.get("message", {}).get("content", "")

        # Default: OpenAI Chat Completions backend.
        response = get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=temperature,
            max_tokens=500
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error calling LLM: {str(e)}"

def _extract_message_text(message: Dict) -> str:
    text = (message.get("text") or "").strip()
    if text:
        return text

    content = message.get("content") or []
    parts = []
    for item in content:
        if item.get("type") == "text" and item.get("text"):
            parts.append(item["text"])

    return "\n".join(parts).strip()

def import_conversations(path: str, memory: HybridMemorySystem, user_id: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a top-level JSON list of conversations.")

    conv_count = 0
    msg_total = 0
    msg_imported = 0
    msg_skipped = 0

    for conv in data:
        conv_count += 1
        messages = conv.get("chat_messages") or []
        if not messages:
            continue

        messages_sorted = sorted(messages, key=lambda m: m.get("created_at") or "")
        for msg in messages_sorted:
            msg_total += 1
            sender = msg.get("sender")
            if sender == "human":
                role = "user"
            elif sender == "assistant":
                role = "assistant"
            else:
                msg_skipped += 1
                continue

            text = _extract_message_text(msg)
            if not text:
                msg_skipped += 1
                continue

            timestamp = msg.get("created_at")
            memory.store_interaction(user_id, role, text, timestamp=timestamp)
            msg_imported += 1

    return {
        "conversations": conv_count,
        "messages_total": msg_total,
        "messages_imported": msg_imported,
        "messages_skipped": msg_skipped,
    }

def print_stats(memory: HybridMemorySystem, user_id: str):
    conn = sqlite3.connect(memory.db_path)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM conversation_turns")
    total_turns = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT user_id) FROM conversation_turns")
    distinct_users = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM conversation_turns WHERE user_id = ?", (user_id,))
    user_turns = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM conversation_turns WHERE user_id = ? AND role = 'user'", (user_id,))
    user_user_turns = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM conversation_turns WHERE user_id = ? AND role = 'assistant'", (user_id,))
    user_assistant_turns = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM explicit_memories WHERE user_id = ?", (user_id,))
    explicit_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM implicit_memories WHERE user_id = ?", (user_id,))
    implicit_count = c.fetchone()[0]

    c.execute("SELECT MIN(timestamp), MAX(timestamp) FROM conversation_turns WHERE user_id = ?", (user_id,))
    first_ts, last_ts = c.fetchone()

    conn.close()

    print("\n📊 Stats")
    print(f"  Total turns (all users): {total_turns}")
    print(f"  Distinct users: {distinct_users}")
    print(f"  Current user_id: {user_id}")
    print(f"  Turns (user): {user_turns} (user: {user_user_turns}, assistant: {user_assistant_turns})")
    print(f"  Explicit memories: {explicit_count}")
    print(f"  Implicit memories: {implicit_count}")
    if first_ts and last_ts:
        print(f"  Time range: {first_ts} → {last_ts}")

def print_explain(memory: HybridMemorySystem, user_id: str, query: str):
    context = memory.explain_context(user_id, query)
    explicit = context.get("explicit_facts", [])
    implicit = context.get("behavioral_patterns", [])
    debug = context.get("debug", {})

    print("\n🧭 Explain")
    print(f"  Query: {query}")
    if debug.get("embedding_backend"):
        print(f"  Embedding backend: {debug['embedding_backend']}")

    print("\n  Explicit facts used:")
    if explicit:
        for mem in explicit:
            src = mem.get("source", "unknown")
            rel = mem.get("relevance", 0.0)
            print(f"    - ({src}, rel={rel:.3f}) {mem.get('content')}")
    else:
        print("    - (none)")

    print("\n  Implicit patterns used:")
    if implicit:
        for pat in implicit:
            conf = pat.get("confidence", 0.0)
            rel = pat.get("relevance", 0.0)
            age = pat.get("age_days", 0)
            category = pat.get("category", "unknown")
            print(f"    - ({category}, conf={conf:.2f}, rel={rel:.3f}, age={age}d) {pat.get('pattern')}")
    else:
        print("    - (none)")

    if debug:
        print("\n  Thresholds:")
        print(f"    - explicit_top_k: {debug.get('explicit_top_k')}")
        print(f"    - recent_explicit_limit: {debug.get('recent_explicit_limit')}")
        print(f"    - implicit_conf_threshold: {debug.get('implicit_conf_threshold')}")
        print(f"    - implicit_relevance_threshold: {debug.get('implicit_relevance_threshold')}")
        print(f"    - implicit_high_conf_bypass: {debug.get('implicit_high_conf_bypass')}")
        print(f"    - explicit_candidates: {debug.get('explicit_candidates')}")
        print(f"    - implicit_candidates: {debug.get('implicit_candidates')}")

def interactive_chat():
    """Run an interactive chat session with memory"""
    print("🧠 Hybrid Memory System - Interactive Mode")
    print("Commands: /help for a list of commands")
    print("-" * 50)

    # Ensure database directory exists
    db_dir = "data"
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "chat_memory.db")

    # Initialize memory system
    memory = HybridMemorySystem(db_path)

    # Simple user ID (in real app, this comes from auth)
    user_id = "interactive_user"

#    # Simulated assistant responses (replace with real LLM API calls)
#    def generate_response(query: str, context: str) -> str:
#        """Mock LLM response - replace with OpenAI/Anthropic/Local"""
#        if context:
#            return f"[Using context] I recall you mentioned: {context[:100]}... Let me help with: {query[:50]}"
#        else:
#            return f"[No context yet] Processing: {query[:50]}"

    conversation_active = True

    while conversation_active:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in {"/help", "help", "?", "/?"}:
                print("\nAvailable commands:")
                print("  /help            Show this help message")
                print("  /quit            Save the current session and exit")
                print("  /memories        Show explicit memories and implicit patterns for this user")
                print("  /forget <text>   Forget explicit memories whose content contains <text>")
                print("  /import <file>   Import chat sessions from a JSON export")
                print("  /stats           Show database stats for the current user")
                print("  /explain <text>  Explain which memories were used for a query")
                print("  /save            Snapshot the current memory DB to data/chat_memory_YYYYMMDD_HHMMSS.db")
                print("  /load <file>     Load a saved DB file into the active session")
                print("  (Anything else)  Is treated as a normal message to the assistant")
                print("\nHow to teach me explicit memories:")
                print("  - 'my name is Hunter'")
                print("  - 'remember that my dog is Enola'")
                print("  - 'remember I have two dogs named Enola and Glacier'")
                print("  - 'I have two dogs: one named Enola, and the other named Glacier'")
                print("  - 'don't forget that I hate spicy food'")
                print("  - 'I work at OpenAI' or 'I'm a software engineer'")
                print("These patterns are stored as explicit memories and surfaced under [Known Facts]/[Observed Preferences].")
                continue

            if user_input.lower() == "/quit":
                print("Saving session...")
                break

            elif user_input.lower() == "/memories":
                # Display current explicit memories
                conn = sqlite3.connect(memory.db_path)
                c = conn.cursor()
                print("\n📋 Explicit Memories:")
                c.execute("SELECT memory_type, content FROM explicit_memories WHERE user_id = ?", (user_id,))
                for row in c.fetchall():
                    print(f"  [{row[0]}] {row[1]}")

                print("\n🔍 Implicit Patterns:")
                c.execute("SELECT category, pattern, confidence FROM implicit_memories WHERE user_id = ?", (user_id,))
                for row in c.fetchall():
                    print(f"  [{row[0]}] {row[1]} (conf: {row[2]:.2f})")
                conn.close()
                continue

            elif user_input.lower().startswith("/forget "):
                # Delete specific memory
                to_forget = user_input[8:]
                conn = sqlite3.connect(memory.db_path)
                c = conn.cursor()
                c.execute("DELETE FROM explicit_memories WHERE user_id = ? AND content LIKE ?",
                         (user_id, f"%{to_forget}%"))
                deleted = c.rowcount
                conn.commit()
                conn.close()
                print(f"Forgot {deleted} memory/ies matching '{to_forget}'")
                continue

            elif user_input.lower().startswith("/import"):
                parts = shlex.split(user_input)
                if len(parts) < 2:
                    print("Usage: /import <file> [--user <user_id>]")
                    continue

                path = parts[1]
                import_user_id = user_id
                if "--user" in parts:
                    idx = parts.index("--user")
                    if idx + 1 < len(parts):
                        import_user_id = parts[idx + 1]

                if not os.path.exists(path):
                    print(f"File not found: {path}")
                    continue

                try:
                    stats = import_conversations(path, memory, import_user_id)
                    print(
                        f"Imported {stats['messages_imported']} / {stats['messages_total']} messages "
                        f"from {stats['conversations']} conversations for user_id={import_user_id} "
                        f"(skipped {stats['messages_skipped']})."
                    )
                except Exception as e:
                    print(f"Error importing conversations: {e}")
                continue

            elif user_input.lower() == "/stats":
                print_stats(memory, user_id)
                continue

            elif user_input.lower().startswith("/explain"):
                query = user_input[len("/explain"):].strip()
                if not query:
                    print("Usage: /explain <text>")
                    continue
                print_explain(memory, user_id, query)
                continue

            elif user_input.lower() == "/save":
                # Save a snapshot of the current SQLite database to a timestamped file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(db_dir, f"chat_memory_{timestamp}.db")
                try:
                    shutil.copy(memory.db_path, filename)
                    print(f"Saved memory database to {filename}")
                except Exception as e:
                    print(f"Error saving database: {e}")
                continue

            elif user_input.lower().startswith("/load"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    print("Usage: /load <file>")
                    continue

                filename = parts[1].strip()
                # If a bare filename is given, look inside the db_dir by default
                if not os.path.isabs(filename) and not os.path.dirname(filename):
                    filename = os.path.join(db_dir, filename)

                if not os.path.exists(filename):
                    print(f"File not found: {filename}")
                    continue

                try:
                    shutil.copy(filename, memory.db_path)
                    # Reinitialize memory system so future calls use the loaded DB
                    memory = HybridMemorySystem(memory.db_path)
                    print(f"Loaded memory database from {filename}")
                except Exception as e:
                    print(f"Error loading database: {e}")
                continue


            # Fast-path: if this is a pure remember command, handle it directly
            # without invoking the LLM. This is faster and avoids confusing the model.
            if memory.is_remember_command(user_input):
                response = memory.process_remember_command(user_id, user_input)
                print(f"Assistant: {response}")
                continue
            # Normal conversation flow
            # 1. Store user message
            memory.store_interaction(user_id, "user", user_input)

            # 2. Retrieve context
            context = memory.get_context(user_id, user_input)
            context_text = memory.format_for_prompt(context)

            # 3. Generate response (replace this with your actual LLM call)
            response = generate_response(user_input, context_text)

            # 4. Store assistant response
            memory.store_interaction(user_id, "assistant", response)

            # 5. Display
            print(f"Assistant: {response}")

            # Optional: Show what context was injected (debugging)
            if context_text:
                print(f"\n[Debug - Context injected]")
                print(context_text[:200] + "..." if len(context_text) > 200 else context_text)

        except KeyboardInterrupt:
            print("\nUse /quit to exit properly")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    interactive_chat()
