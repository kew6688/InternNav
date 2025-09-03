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

    proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


if __name__ == '__main__':
    try:
        start_server()
        time.sleep(5)
    except Exception as e:
        print(f'exception is {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)
