'''
Test the evaluator eval logic without model involve.
The main progress:
    Init => warm up => fake one action
'''


def main():
    from enum import Enum

    from internnav.configs.agent import AgentCfg
    from internnav.configs.evaluator import (
        EnvCfg,
        EvalCfg,
        EvalDatasetCfg,
        SceneCfg,
        TaskCfg,
    )

    class runner_status_code(Enum):
        NORMAL = 0
        WARM_UP = 1
        NOT_RESET = 3
        TERMINATED = 2
        STOP = 4

    eval_cfg = EvalCfg(
        agent=AgentCfg(
            server_port=8087,
            model_name='rdp',
            ckpt_path='checkpoints/r2r/fine_tuned/rdp',
            model_settings={},
        ),
        env=EnvCfg(
            env_type='vln_pe',
            env_settings={
                'use_fabric': False,
                'headless': True,  # display option: set to False will open isaac-sim interactive window
            },
        ),
        task=TaskCfg(
            task_name='test_evaluator',
            task_settings={
                'env_num': 2,
                'use_distributed': False,  # Ray distributed framework
                'proc_num': 8,
            },
            scene=SceneCfg(
                scene_type='mp3d',
                scene_data_dir='data/scene_data/mp3d_pe',
            ),
            robot_name='h1',
            robot_usd_path='data/Embodiments/vln-pe/h1/h1_vln_pointcloud.usd',
            camera_resolution=[256, 256],  # (W,H)
            camera_prim_path='torso_link/h1_pano_camera_0',
        ),
        dataset=EvalDatasetCfg(
            dataset_type="mp3d",
            dataset_settings={
                'base_data_dir': 'data/vln_pe/raw_data/r2r',
                'split_data_types': ['val_unseen', 'val_seen'],
                'filter_stairs': False,
            },
        ),
        eval_settings={'save_to_json': False, 'vis_output': False},  # save result to video under logs/
    )
    print(eval_cfg)

    # cfg = get_config(eval_cfg)
    # try:
    #     evaluator = Evaluator.init(cfg)
    # except Exception as e:
    #     print(e)

    # print('--- VlnPeEvaluator start ---')
    # obs, reset_info = evaluator.env.reset()
    # for info in reset_info:
    #     if info is None:
    #         continue
    #     progress_log_multi_util.trace_start(
    #         trajectory_id=evaluator.now_path_key(info),
    #     )

    # obs = evaluator.warm_up()
    # evaluator.fake_obs = obs[0][evaluator.robot_name]
    # action = [{evaluator.robot_name: {'stand_still': []}} for _ in range(evaluator.env_num * evaluator.proc_num)]
    # obs = evaluator._obs_remove_robot_name(obs)
    # evaluator.runner_status = np.full(
    #     (evaluator.env_num * evaluator.proc_num),
    #     runner_status_code.NORMAL,
    #     runner_status_code,
    # )
    # evaluator.runner_status[[info is None for info in reset_info]] = runner_status_code.TERMINATED

    # while evaluator.env.is_running():

    #     obs, terminated = evaluator.env_step(action)
    #     break

    # evaluator.env.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'exception is {e}')
        import sys
        import traceback

        traceback.print_exc()
        sys.exit(1)
