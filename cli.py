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

    system_prompt = """You are a helpful assistant with memory of past conversations."""
    
    if context:
        system_prompt += f"\n\nRelevant context about the user:\n{context}"
    
    try:
        response = get_client().chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" if you don't have GPT-4 access
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

def interactive_chat():
    """Run an interactive chat session with memory"""
    print("🧠 Hybrid Memory System - Interactive Mode")
    print("Commands: /quit, /memories, /forget <text>, /save, /load <file>")
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
