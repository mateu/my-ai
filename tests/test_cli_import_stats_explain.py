import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_import_stats_explain(tmp_path):
    """Verify /import, /stats, and /explain commands in the CLI."""
    project_root = Path(__file__).resolve().parents[1]
    workdir = tmp_path

    # Copy cli.py
    src_cli = project_root / "cli.py"
    dst_cli = workdir / "cli.py"
    dst_cli.write_text(src_cli.read_text())

    # Copy the my_ai package directory recursively so imports still work.
    src_pkg = project_root / "my_ai"
    dst_pkg = workdir / "my_ai"
    dst_pkg.mkdir(parents=True, exist_ok=True)
    for sub in src_pkg.rglob("*"):
        if "__pycache__" in sub.parts:
            continue
        rel = sub.relative_to(src_pkg)
        target = dst_pkg / rel
        if sub.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.write_text(sub.read_text())

    # Create a small conversations export for import.
    conversations = [
        {
            "uuid": "conv-1",
            "name": "Test import",
            "summary": "",
            "created_at": "2026-01-01T10:00:00Z",
            "updated_at": "2026-01-01T10:00:10Z",
            "account": {"uuid": "acct-1"},
            "chat_messages": [
                {
                    "uuid": "msg-1",
                    "text": "my name is Alice",
                    "content": [
                        {
                            "type": "text",
                            "text": "my name is Alice",
                            "citations": [],
                            "start_timestamp": "2026-01-01T10:00:01Z",
                            "stop_timestamp": "2026-01-01T10:00:01Z",
                            "flags": None,
                        }
                    ],
                    "sender": "human",
                    "created_at": "2026-01-01T10:00:01Z",
                    "updated_at": "2026-01-01T10:00:01Z",
                    "attachments": [],
                    "files": [],
                },
                {
                    "uuid": "msg-2",
                    "text": "Nice to meet you, Alice.",
                    "content": [
                        {
                            "type": "text",
                            "text": "Nice to meet you, Alice.",
                            "citations": [],
                            "start_timestamp": "2026-01-01T10:00:05Z",
                            "stop_timestamp": "2026-01-01T10:00:05Z",
                            "flags": None,
                        }
                    ],
                    "sender": "assistant",
                    "created_at": "2026-01-01T10:00:05Z",
                    "updated_at": "2026-01-01T10:00:05Z",
                    "attachments": [],
                    "files": [],
                },
            ],
        }
    ]

    export_path = workdir / "conversations-7.json"
    export_path.write_text(json.dumps(conversations, ensure_ascii=False, indent=2))

    env = os.environ.copy()
    env["AI_MEMORY_OFFLINE"] = "1"

    session_script = f"""/import {export_path}
/stats
/explain what is my name?
/quit
"""

    proc = subprocess.run(
        [sys.executable, "cli.py"],
        input=session_script,
        text=True,
        capture_output=True,
        cwd=str(workdir),
        env=env,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "Imported " in proc.stdout
    assert "📊 Stats" in proc.stdout
    assert "🧭 Explain" in proc.stdout
    assert "Explicit facts used:" in proc.stdout
    assert "name: Alice" in proc.stdout
