import numpy as np
import mujoco_py
import gym
from typing import Union
import time
from os import path


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

    def __init__(self, model, **kwargs):
        self.width, self.height = 480, 480
        # self.model = mujoco_py.load_model_from_path(self.fullpath)
        self.model = mujoco_py.load_model_from_xml(model)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self._viewers = {}
        self.viewer = None
        self.camera_name = "camera"
        self.camera_id = None # 2
        

    def seed(self, seed):
        # super().reset(seed=seed) # TODO: seed, gym version
        np.random.seed(seed)

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        state = self.sim.get_state()
        state = mujoco_py.MjSimState(state.time, qpos, qvel, state.act, state.udd_state)
        self.sim.set_state(state)
        self.sim.forward()

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
            # time.sleep(wait_time)
