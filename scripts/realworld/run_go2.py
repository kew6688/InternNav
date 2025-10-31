# apps/run_go2.py
import rclpy

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, TaskCfg
from internnav.env.base import Env


def main():
    # ROS init
    rclpy.init()

    # ---- Build configs ----
    agent_cfg = AgentCfg(
        server_host="127.0.0.1",
        server_port=8087,
        model_name="internvla_n1_realworld",
        model_settings={
            'policy_name': 'InternVLAN1_Policy',
            'state_encoder': None,
        },
    )

    env_cfg = EnvCfg(env_type="realworld_go2")
    task_cfg = TaskCfg(
        instruction="Turn around and go to the red bin, then enter the open door on the right and stop at the monitor.",
        camera_intrinsic=None,  # use default in env if None
        agent_cfg=agent_cfg,
    )

    # ---- Init Env ----
    env = Env.init(env_cfg, task_cfg)
    try:
        obs, info = env.reset()
        # Real-world loop is thread-driven; just spin ROS.
        rclpy.spin(env._node)  # expose if you prefer a small wrapper method
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
