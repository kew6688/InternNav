"""
Test if the server start successfully and exit successfully
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
    # No start_new_session，stay with parent process
    proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,  # avoid using PIPE, overflow the process buffer
        stderr=subprocess.STDOUT,
    )
    return proc


if __name__ == '__main__':
    proc = None
    try:
        proc = start_server()
        time.sleep(3)
    except Exception as e:
        print(f'exception is {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)
    # finally:
    #     if proc and proc.poll() is None:
    #         print("Shutting down server...")
    #         proc.terminate()
    #         try:
    #             proc.wait(timeout=10)
    #         except subprocess.TimeoutExpired:
    #             print("Force killing server...")
    #             proc.kill()
