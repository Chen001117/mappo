import numpy as np
import time
from os import path
import mujoco_py
import gym
from gym import utils
from onpolicy.envs.mujoco.mujoco_env import MuJocoPyEnv
from gym.spaces import Box, Tuple
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
        self.width, self.height = 480, 480
        self.fullpath = path.join(path.dirname(__file__), "assets", "navigation.xml")
        self.model = mujoco_py.load_model_from_path(self.fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
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
            time.sleep(wait_time)


class NavigationEnv(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.img_size = 60
        # self.observation_space = Tuple((
        #     Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64),
        #     Box(low=-np.inf, high=np.inf, shape=(3,self.img_size,self.img_size), dtype=np.float64),
        # ))
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float64)
        aspace_low = np.array([-0.6, -0.6, -0.6])
        aspace_high = np.array([0.6, 0.6, 0.6])
        self.action_space = Box(
            low=aspace_low, high=aspace_high, shape=(3,), dtype=np.float64
        )
        self.kp = np.array([50, 50, 50])
        self.kd = np.array([0.01, 0.01, 0.01])
        self.goal = np.zeros(2)
        self.his_vel = []
        self.t = 0.

    def reset(self):
        regenerate = True
        while regenerate:
            self.t = 0.
            super().reset()
            self.reset_model()
            regenerate = self._get_done()
            self.prev_output = np.zeros(3)
        for _ in range(10):
            self.do_simulation(np.zeros(3), self.frame_skip)
        self.t = 0.
        position = self.sim.data.qpos.copy().flat[:2]
        self.prev_dist = np.linalg.norm(position-self.goal)
        obs = self._get_obs()
        return obs, {}

    def reset_model(self):
        self.sim.reset()
        pos_x = np.random.rand() * 9. - 4.5
        pos_y = np.random.rand() * 9. - 4.5
        yaw = np.random.rand() * np.pi * 2.
        self.goal = np.random.rand(2) * 9. - 4.5
        qpos = [pos_x, pos_y, yaw, 0.3]
        self.osize = 20
        self.obs_num = 20
        self.rands = np.random.choice(range(self.osize*self.osize), self.obs_num, replace=False)
        # self.obs_map = np.zeros([3, self.img_size, self.img_size])
        self.obs_list = []
        for rand in self.rands:
            pos_x = (rand%self.osize) * 0.5 - 4.75 
            pos_y = (rand//self.osize) * 0.5 - 4.75 
            qpos.append(pos_x)
            qpos.append(pos_y)
            # pos_x1 = (rand%self.osize) * (self.img_size//self.osize)
            # pos_x2 = (rand%self.osize+1) * (self.img_size//self.osize)
            # pos_y1 = (rand//self.osize) * (self.img_size//self.osize)
            # pos_y2 = (rand//self.osize+1) * (self.img_size//self.osize)
            # self.obs_map[0, pos_x1:pos_x2, pos_y1:pos_y2] = 1.
            self.obs_list.append([pos_x, pos_y])
        self.obs_list = np.array(self.obs_list)
        qpos.append(self.goal[0])        
        qpos.append(self.goal[1])  
        self.set_state(np.array(qpos), np.zeros_like(qpos))

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()[:2]
        dir_cos = np.cos(self.sim.data.qpos.flat.copy()[2:3])
        dir_sin = np.sin(self.sim.data.qpos.flat.copy()[2:3])
        velocity = self.sim.data.qvel.flat.copy()[:3]
        observation = np.concatenate([
            position, dir_cos, dir_sin, velocity, self.goal.copy(), [self.t]
        ]).ravel()

        lnum = 12
        d = np.ones(lnum) * 4.
        for i in range(lnum):
            dir_vec = np.array([np.cos(i*np.pi*2/lnum), np.sin(i*np.pi*2/lnum)])
            for j in range(40):
                pnt = position + dir_vec * j / 10.
                dist = pnt - self.obs_list
                dist = np.linalg.norm(dist, np.inf, axis=-1)
                if min(dist) < 0.25:
                    d[i] = j / 10.
                    break
                if (pnt<-5).any() or (pnt>5.).any():
                    d[i] = j / 10.
                    break
        observation = np.concatenate([observation, d])

        # obs_map = self.obs_map.copy()
        # pos_x = int((position[0]+5.) * (self.img_size/10.))
        # pos_y = int((position[1]+5.) * (self.img_size/10.))
        # obs_map[1, pos_x, pos_y] = 1.
        # pos_x = int((self.goal[0]+5.) * (self.img_size/10.))
        # pos_x = int((self.goal[1]+5.) * (self.img_size/10.))
        # obs_map[2, pos_x, pos_y] = 1.
        return observation#, obs_map

    def _local_to_global(self, input_action):
        local_action = input_action[:2].copy().reshape([1,2])
        theta = self.sim.data.qpos.flat.copy()[2]
        rotate_mat = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        global_action = local_action[:2] @ rotate_mat
        output_action = np.concatenate([global_action[0], input_action[2:].copy()])
        return output_action

    def _pd_contorl(self, target_vel, dt):
        error = target_vel - self.sim.data.qvel[:3]
        d_output = self.sim.data.qvel[:3] - self.prev_output
        torque = self.kp*error-self.kd*(d_output/dt)
        torque = np.clip(torque, self.torque_low, self.torque_high)
        self.prev_output = torque.copy()
        return torque

    def _get_reward(self):
        position = self.sim.data.qpos.copy().flat[:2]
        dist = np.linalg.norm(position-self.goal)
        rew = self.prev_dist - dist #+ 0.01
        rew = 0.1 if dist < 0.25 else rew 
        rew -= self.con_done * 2.5
        self.prev_dist = dist.copy()
        return rew

    def _get_done(self):
        done = self.t > 10.
        self.con_done = False
        for i in range(self.sim.data.ncon):
            con = self.sim.data.contact[i]
            obj1 = self.sim.model.geom_id2name(con.geom1)
            obj2 = self.sim.model.geom_id2name(con.geom2)
            self.con_done |= (obj1=="dog" or obj2=="dog") \
                and (obj1!="floor" and obj2!="floor" and obj1!="destination" and obj2!="destination")
        
        return done or self.con_done

    def do_simulation(self, action, n_frames):
        for _ in range(n_frames):
            ctrl = self._pd_contorl(action, self.dt/n_frames)
            assert np.array(ctrl).shape==self.action_space.shape, "Action dimension mismatch"
            self.sim.data.ctrl[:] = ctrl
            self.sim.step()
        self.t += self.dt

    def step(self, command):
        command = np.clip(command, self.action_space.low, self.action_space.high)
        # action = self._local_to_global(command)
        action = command.copy()
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        terminated = self._get_done()
        reward = self._get_reward()
        info = dict()
        return observation, reward, terminated, False, info

        