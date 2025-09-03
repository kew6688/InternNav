"""
Test if the server starts successfully and is still alive after sleep.
"""
import subprocess
import sys
import time


def start_server():
    server_cmd = [
        sys.executable,
        "internnav/agent/utils/server.py",
        "--config",
        "scripts/eval/configs/challenge_cfg.py",
    ]

    proc = subprocess.Popen(
        server_cmd,
        stdout=None,
        stderr=None,
        start_new_session=True,
    )
    return proc


if __name__ == '__main__':
    try:
        proc = start_server()
        time.sleep(5)

        # Raise if process exited
        if proc.poll() is not None:
            raise RuntimeError(f"❌ Server exited too early with code {proc.returncode}")
        print("✅ Server is still alive after 5 seconds.")

        if proc and proc.poll() is None:
            print("Shutting down server...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                raise RuntimeError("❌ Server failed to shut down within 10 seconds.")

    except Exception as e:
        print(f'exception is {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)
