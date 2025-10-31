# internnav/envs/go2_env.py
from __future__ import annotations

import io
import math
import threading
import time
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

# user-provided utilities/controllers
from controllers import Mpc_controller, PID_controller
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from PIL import Image as PIL_Image
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from thread_utils import ReadWriteLock

from internnav.agent.client import AgentClient  # <- your AgentClient
from internnav.configs.agent import (  # types only, StepRequest used by AgentClient internals
    AgentCfg,
    StepRequest,
)
from internnav.configs.evaluator import EnvCfg, TaskCfg
from internnav.env.base import Env  # <- this is your improved Env base from earlier


class ControlMode(Enum):
    PID = 1
    MPC = 2


@Env.register("realworld_go2")
class Go2Env(Env):
    """
    Real-world GO2 Env that:
    - Subscribes to RGB-D + Odom
    - Packages observations and queries Agent service via AgentClient
    - Switches between MPC (trajectory) and PID (discrete actions)
    - Publishes Twist to /cmd_vel_bridge
    """

    def __init__(self, env_config: EnvCfg, task_config: TaskCfg, *, render_mode: Optional[str] = None):
        super().__init__(env_config, task_config, render_mode=render_mode)

        # ---- ROS node ----
        self._node = Node("go2_env_manager")

        # Subscriptions (RGB + depth sync, Odom)
        rgb_sub = Subscriber(self._node, Image, "/camera/camera/color/image_raw")
        depth_sub = Subscriber(self._node, Image, "/camera/camera/aligned_depth_to_color/image_raw")
        self._sync = ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=1, slop=0.1)
        self._sync.registerCallback(self._rgb_depth_cb)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self._odom_sub = self._node.create_subscription(Odometry, "/odom_bridge", self._odom_cb, qos)

        # Publisher
        self._cmd_pub = self._node.create_publisher(Twist, "/cmd_vel_bridge", 5)

        # ---- State / buffers ----
        self._cv_bridge = CvBridge()
        self._rgb_image = None  # np.ndarray HxWx3 uint8
        self._rgb_bytes = None  # BytesIO (JPEG)
        self._depth_image = None  # np.ndarray HxW float32 (meters)
        self._depth_bytes = None  # BytesIO (PNG-16U in mm*10)
        self._rgb_time = 0.0
        self._depth_time = 0.0

        self._odom = None  # [x, y, yaw]
        self._odom_queue = deque(maxlen=50)  # [(stamp, [x,y,yaw])]
        self._odom_stamp_wall = 0.0
        self._lin_vel = 0.0
        self._ang_vel = 0.0

        # Homogeneous frames for PID
        self._H_odom = None  # 4x4
        self._H_goal = None  # 4x4
        self._vel_pair = None  # [v, w]

        # Locks
        self._rgbd_lock = ReadWriteLock()
        self._odom_lock = ReadWriteLock()
        self._mpc_lock = ReadWriteLock()

        # Control/Planning
        self._pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=0.6, max_w=0.5)
        self._mpc: Optional[Mpc_controller] = None
        self._mode = ControlMode.MPC

        self._desired_v = 0.0
        self._desired_w = 0.0

        self._new_image_arrived = False

        # Task / instruction & camera intrinsics (can be part of TaskCfg)
        self._instruction = getattr(task_config, "instruction", "Go to the target")
        self._K = getattr(
            task_config,
            "camera_intrinsic",
            np.array(
                [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
        )

        # Agent client
        agent_cfg: AgentCfg = getattr(task_config, "agent_cfg")  # provided by caller
        self._agent = AgentClient(agent_cfg)  # creates/initializes on server, returns agent_name

        # Threads
        self._stop_evt = threading.Event()
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._plan_thread = threading.Thread(target=self._plan_loop, daemon=True)

        # Book-keeping
        self._frame_cache: Dict[int, Dict[str, Any]] = {}
        self._http_idx = -1
        self._first_run_wall = 0.0

        # start threads after init
        self._control_thread.start()
        self._plan_thread.start()

    # ----------------- Env API -----------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None):
        # Reset the remote agent session
        # NOTE: you can add a DELETE /agent/{id} + re-init if you need a hard reset
        self._http_idx = -1
        self._first_run_wall = 0.0
        self._mode = ControlMode.MPC
        if self._H_odom is not None:
            self._H_goal = self._H_odom.copy()

        # Return current obs immediately (or wait until first RGB-D arrives)
        obs = self.get_observation()
        info = self.get_info()
        return obs, info

    def step(self, action: Any):
        """
        For real-world loop, actions are produced internally (from Agent).
        This method can be a no-op or emit the last applied action and new obs.
        """
        obs = self.get_observation()
        # reward/terminations are domain-specific; return 0/False/False
        return obs, 0.0, False, False, self.get_info()

    def render(self):
        return None  # you can add MJPEG or last RGB here

    def get_observation(self) -> Mapping[str, Any]:
        """
        A snapshot of the latest observation (RGB, Depth, Odometry).
        """
        with self._with_read(self._rgbd_lock), self._with_read(self._odom_lock):
            obs = {
                "rgb": None if self._rgb_image is None else np.asarray(self._rgb_image, copy=True),
                "depth": None if self._depth_image is None else np.asarray(self._depth_image, copy=True),
                "odom": None if self._odom is None else list(self._odom),
                "intrinsic": self._K.copy(),
                "instruction": self._instruction,
                "stamp": self._rgb_time,
            }
        return obs

    def get_info(self) -> Mapping[str, Any]:
        info = {
            "mode": self._mode.name,
            "desired_vw": (self._desired_v, self._desired_w),
        }
        return info

    def close(self) -> None:
        if self._closed:
            return
        self._stop_evt.set()
        # best-effort: let threads exit
        for t in (self._control_thread, self._plan_thread):
            if t.is_alive():
                t.join(timeout=1.0)
        try:
            self._node.destroy_node()
        except Exception:
            pass
        super().close()

    # ----------------- ROS Callbacks -----------------

    def _rgb_depth_cb(self, rgb_msg: Image, depth_msg: Image):
        # RGB -> JPEG
        raw_rgb = self._cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
        rgb_img = PIL_Image.fromarray(raw_rgb)
        rgb_bytes = io.BytesIO()
        rgb_img.save(rgb_bytes, format='JPEG')
        rgb_bytes.seek(0)

        # depth 16UC1 mm (aligned) -> float32 meters (clamped), then encode 16U PNG with scale 10000
        raw_depth = self._cv_bridge.imgmsg_to_cv2(depth_msg, '16UC1').astype(np.float32)
        raw_depth[np.isnan(raw_depth)] = 0
        raw_depth[np.isinf(raw_depth)] = 0
        depth_m = raw_depth / 1000.0
        depth_m[depth_m < 0] = 0
        depth_enc = (np.clip(depth_m * 10000.0, 0, 65535)).astype(np.uint16)
        depth_png = PIL_Image.fromarray(depth_enc)
        depth_bytes = io.BytesIO()
        depth_png.save(depth_bytes, format='PNG')
        depth_bytes.seek(0)

        with self._with_write(self._rgbd_lock):
            self._rgb_image = raw_rgb
            self._rgb_bytes = rgb_bytes
            self._rgb_time = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec / 1e9

            self._depth_image = depth_m
            self._depth_bytes = depth_bytes
            self._depth_time = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec / 1e9

        self._new_image_arrived = True

    def _odom_cb(self, msg: Odometry):
        zz = msg.pose.pose.orientation.z
        ww = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * zz * ww, 1 - 2 * zz * zz)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z

        with self._with_write(self._odom_lock):
            self._odom = [x, y, yaw]
            self._odom_queue.append((time.time(), self._odom.copy()))
            self._odom_stamp_wall = time.time()
            self._lin_vel = v
            self._ang_vel = w

            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=np.float32)
            self._H_odom = np.eye(4, dtype=np.float32)
            self._H_odom[:2, :2] = R
            self._H_odom[:2, 3] = [x, y]
            self._vel_pair = [v, w]
            if self._H_goal is None:
                self._H_goal = self._H_odom.copy()

    # ----------------- Control/Planning Threads -----------------

    def _control_loop(self):
        rate_sec = 0.1
        while not self._stop_evt.is_set():
            mode = self._mode
            if mode == ControlMode.MPC:
                odom = self._copy_odom()
                if odom is not None:
                    with self._with_read(self._mpc_lock):
                        if self._mpc is not None:
                            try:
                                u, _ = self._mpc.solve(np.array(odom))
                                v, w = float(u[0, 0]), float(u[0, 1])
                            except Exception:
                                v, w = 0.0, 0.0
                            self._desired_v, self._desired_w = v, w
                            self._publish_cmd(v, w)
            else:  # PID
                H_odom, H_goal, vel = self._H_odom, self._H_goal, self._vel_pair
                if H_odom is not None and H_goal is not None and vel is not None:
                    v, w, _, _ = self._pid.solve(H_odom, H_goal, vel)
                    if v < 0:
                        v = 0.0
                    self._desired_v, self._desired_w = float(v), float(w)
                    self._publish_cmd(v, w)
            time.sleep(rate_sec)

    def _plan_loop(self):
        DESIRED_PERIOD = 0.3
        while not self._stop_evt.is_set():
            t0 = time.time()
            if not self._new_image_arrived:
                time.sleep(0.01)
                continue
            self._new_image_arrived = False

            # Snapshot latest rgb/depth + closest odom in time
            rgb_bytes, depth_bytes, rgb_stamp = self._copy_rgbd()
            odom_match = self._closest_odom(rgb_stamp)
            if rgb_bytes is None or depth_bytes is None or odom_match is None:
                # Not ready yet
                time.sleep(0.05)
                continue

            # Build observation for Agent
            # We keep your JPEG/PNG bytes for faithful reproduction on the server (Agent will deserialize).
            obs: List[Dict[str, Any]] = [
                {
                    "rgb_bytes": rgb_bytes.getvalue(),  # raw bytes (will be pickled+base64 by AgentClient)
                    "depth_bytes": depth_bytes.getvalue(),
                    "camera_pose": np.eye(4, dtype=np.float32),
                    "instruction": self._instruction,
                    "intrinsic": self._K,
                    "odom": odom_match,  # [x, y, yaw] at capture time
                }
            ]

            try:
                # -> Agent service
                action_or_traj = self._agent.step(obs)
            except Exception as e:
                # Keep robot safe on RPC failure
                action_or_traj = {"discrete_action": [5]}  # noop/stop as a safe fallback
                print(f"[Agent RPC error] {e}")

            # Interpret agent output
            if "trajectory" in action_or_traj:
                traj_body: List[List[float]] = action_or_traj["trajectory"]
                od = odom_match
                if od is not None and len(traj_body) > 0:
                    yaw = od[2]
                    w_T_b = np.array(
                        [
                            [np.cos(yaw), -np.sin(yaw), 0, od[0]],
                            [np.sin(yaw), np.cos(yaw), 0, od[1]],
                            [0.0, 0.0, 1, 0.0],
                            [0.0, 0.0, 0, 1.0],
                        ],
                        dtype=np.float32,
                    )
                    world_pts = []
                    for i, (tx, ty, *_) in enumerate(traj_body):
                        if i < 3:
                            continue
                        wP = (w_T_b @ np.array([tx, ty, 0.0, 1.0], dtype=np.float32))[:2]
                        world_pts.append(wP)
                    world_pts = np.array(world_pts, dtype=np.float32) if len(world_pts) else None

                    with self._with_write(self._mpc_lock):
                        if world_pts is not None:
                            if self._mpc is None:
                                self._mpc = Mpc_controller(world_pts)
                            else:
                                self._mpc.update_ref_traj(world_pts)
                    self._mode = ControlMode.MPC
            elif "discrete_action" in action_or_traj:
                acts = action_or_traj["discrete_action"]
                if acts not in ([5], [9]):  # your original safe/no-op codes
                    self._incremental_change_goal(acts)
                    self._mode = ControlMode.PID

            # Keep loop period
            dt = time.time() - t0
            if dt < DESIRED_PERIOD:
                time.sleep(DESIRED_PERIOD - dt)

    # ----------------- Helpers -----------------

    def _publish_cmd(self, v: float, w: float):
        msg = Twist()
        msg.linear.x = float(v)
        msg.linear.y = 0.0
        msg.angular.z = float(w)
        self._cmd_pub.publish(msg)

    def _closest_odom(self, stamp: float) -> Optional[List[float]]:
        with self._with_read(self._odom_lock):
            if not self._odom_queue:
                return None
            best = min(self._odom_queue, key=lambda p: abs(p[0] - stamp))
            return best[1].copy()

    def _copy_rgbd(self):
        with self._with_read(self._rgbd_lock):
            return (
                None if self._rgb_bytes is None else io.BytesIO(self._rgb_bytes.getvalue()),
                None if self._depth_bytes is None else io.BytesIO(self._depth_bytes.getvalue()),
                float(self._rgb_time),
            )

    def _copy_odom(self):
        with self._with_read(self._odom_lock):
            return None if self._odom is None else self._odom.copy()

    def _incremental_change_goal(self, actions: List[int]):
        if self._H_goal is None or self._H_odom is None:
            # Initialize
            self._H_goal = self._H_odom.copy() if self._H_odom is not None else np.eye(4, dtype=np.float32)

        H = self._H_goal.copy()
        for a in actions:
            if a == 0:
                continue
            elif a == 1:  # forward
                yaw = math.atan2(H[1, 0], H[0, 0])
                H[0, 3] += 0.25 * math.cos(yaw)
                H[1, 3] += 0.25 * math.sin(yaw)
            elif a == 2:  # left 15deg
                ang = math.radians(15.0)
                R = np.array(
                    [[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1]], dtype=np.float32
                )
                H[:3, :3] = R @ H[:3, :3]
            elif a == 3:  # right 15deg
                ang = -math.radians(15.0)
                R = np.array(
                    [[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1]], dtype=np.float32
                )
                H[:3, :3] = R @ H[:3, :3]
        self._H_goal = H

    # Context-manager helpers for your ReadWriteLock (Python with-statements)
    from contextlib import contextmanager

    @contextmanager
    def _with_read(self, lock: ReadWriteLock):
        lock.acquire_read()
        try:
            yield
        finally:
            lock.release_read()

    @contextmanager
    def _with_write(self, lock: ReadWriteLock):
        lock.acquire_write()
        try:
            yield
        finally:
            lock.release_write()
