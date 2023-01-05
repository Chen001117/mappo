import numpy as np
import time
from os import path
import mujoco_py
import gym
from gym import utils
from onpolicy.envs.mujoco.mujoco_env import MuJocoPyEnv
from gym.spaces import Box
from typing import Optional, Union

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

class BaseEnv(gym.Env):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 125,
    }

    def __init__(self, **kwargs):
        self.fullpath = path.join(path.dirname(__file__), "assets", "navigation.xml")
        self.width, self.height = 480, 480
        self.model = mujoco_py.load_model_from_path(self.fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self._viewers = {}
        self.frame_skip = 4
        self.viewer = None
        self.camera_name = "camera"
        self.camera_id = 2
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        self.torque_low, self.torque_high = bounds.T
        self._reset_noise_scale = 5e-3

    def seed(self, seed):
        super().reset(seed=seed)

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        state = self.sim.get_state()
        state = mujoco_py.MjSimState(state.time, qpos, qvel, state.act, state.udd_state)
        self.sim.set_state(state)
        self.sim.forward()

    def do_simulation(self, ctrl, n_frames):
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        if np.array(ctrl).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()
        self.t += self.dt

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def _get_viewer(
        self, mode
    ) -> Union["mujoco_py.MjViewer", "mujoco_py.MjRenderContextOffscreen"]:
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)

            elif mode in {"rgb_array", "depth_array"}:
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
            else:
                raise AttributeError(
                    f"Unknown mode: {mode}, expected modes: {self.metadata['render_modes']}"
                )

            self.viewer_setup()
            self._viewers[mode] = self.viewer

        return self.viewer

    def render(self, mode):
        if mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        width, height = self.width, self.height
        camera_name, camera_id = self.camera_name, self.camera_id
        if mode in {"rgb_array", "depth_array"}:
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None and camera_name in self.model._camera_name2id:
                if camera_name in self.model._camera_name2id:
                    camera_id = self.model.camera_name2id(camera_name)

                self._get_viewer(mode).render(
                    width, height, camera_id=camera_id
                )

        if mode == "rgb_array":
            data = self._get_viewer(mode).read_pixels(
                width, height, depth=False
            )
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "depth_array":
            self._get_viewer(mode).render(width, height)
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(
                width, height, depth=True
            )[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == "human":
            start = time.time()
            self._get_viewer(mode).render()
            wait_time = max(self.dt-time.time()+start, 0.)
            time.sleep(wait_time)


class NavigationEnv(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
        )
        self.action_space = Box(
            low=-1., high=1., shape=(3,), dtype=np.float64
        )
        self.kp = 50.
        self.kd = 0.01
        self.goal = np.zeros(2)

    def reset(self):
        self.t = 0.
        self.prev_output = np.zeros(3)
        super().reset()
        self.sim.reset()
        self.reset_model()
        obs = self._get_obs()
        self.goal = np.random.rand(2) * 10. - 5.
        return obs, {}

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()[:3]
        velocity = self.sim.data.qvel.flat.copy()[:3]
        observation = np.concatenate((position, velocity, self.goal.copy())).ravel()
        return observation

    def _local_to_global(self, input_action):
        local_action = input_action[:2].copy().reshape([1,2])
        theta = self.sim.data.qpos.flat.copy()[2]
        rotate_mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.cos(theta), np.sin(theta)],
        ])
        global_action = local_action[:2] @ rotate_mat
        output_action = np.concatenate([global_action[0], input_action[2]])
        return output_action

    def _pd_contorl(self, target_vel):
        error = target_vel - self.sim.data.qvel[:3]
        d_output = self.sim.data.qvel[:3] - self.prev_output
        torque = self.kp*error-self.kd*(d_output/self.dt)
        torque = np.clip(torque, self.torque_low, self.torque_high)
        self.prev_output = torque.copy()
        return torque
    
    def _get_reward(self):
        position = self.sim.data.qpos.flat.copy()[:2]
        dist = np.linalg.norm(position-self.goal)
        return np.exp(-dist)

    def _get_done(self):
        return self.t > 10. 

    def step(self, action):
        action = self._local_to_global(action)
        torque = self._pd_contorl(action)
        self.do_simulation(torque, self.frame_skip)
        observation = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_done()
        info = dict()
        return observation, reward, terminated, False, info

        