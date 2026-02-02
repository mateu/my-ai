import os
import subprocess
import sys
from pathlib import Path


def test_cli_end_to_end(tmp_path):
    """Minimal end-to-end test of cli.py.

    - Starts the CLI.
    - Sends a message that should become an explicit memory.
    - Saves the DB with /save.
    - Shows memories with /memories and checks the fact is present.
    - Quits cleanly with /quit.
    """

    # tests/ lives one level below the project root; climb up to find cli.py, etc.
    project_root = Path(__file__).resolve().parents[1]

    # Run the CLI in an isolated temp directory, copying just what we need.
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
        # Skip compiled artifacts; we only need source files for the test.
        if "__pycache__" in sub.parts:
            continue
        rel = sub.relative_to(src_pkg)
        target = dst_pkg / rel
        if sub.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.write_text(sub.read_text())

    env = os.environ.copy()
    # Disable real network calls in generate_response
    env["AI_MEMORY_OFFLINE"] = "1"

    # Script a short session: create a fact, save, inspect, quit.
    session_script = """my name is TestUser
/save
/memories
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

    # Process should exit cleanly.
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    # We should have saved a snapshot and printed explicit memories.
    assert "Saved memory database to" in proc.stdout
    assert "📋 Explicit Memories:" in proc.stdout
    assert "[fact] name: TestUser" in proc.stdout
    assert "Saving session..." in proc.stdout
