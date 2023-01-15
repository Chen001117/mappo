import numpy as np
import time
from os import path
import mujoco_py
import gym
from gym import utils
from onpolicy.envs.mujoco.mujoco_env import MuJocoPyEnv
from onpolicy.envs.mujoco.xml_gen import get_xml
from gym.spaces import Box, Tuple
from typing import Optional, Union
import imageio

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

    def __init__(self, rank, **kwargs):
        self.rank = rank
        self._init()

    def _init(self):
        
        self.width, self.height = 480, 480
        self.fullpath = path.join(path.dirname(__file__), "assets", "navigation.xml")
        self.map = self._get_map()
        # self.model = mujoco_py.load_model_from_path(self.fullpath)
        fullpath = path.join(path.dirname(__file__), "map_{:03d}.png".format(self.rank+500))
        self.model = mujoco_py.load_model_from_xml(get_xml(fullpath))
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

        self.obs_size = 104
        self.con_size = 14
        self.obs_map = np.ones([self.obs_size*2+512,self.obs_size*2+512,1])
        self.con_map = np.ones([self.con_size*2+512,self.con_size*2+512,1])
        self.obs_map[self.obs_size:-self.obs_size,self.obs_size:-self.obs_size] = self.map.copy()
        self.con_map[self.con_size:-self.con_size,self.con_size:-self.con_size] = self.map.copy()
        self.obs_map = self.obs_map.transpose([2,0,1])
        self.con_map = self.con_map.transpose([2,0,1])
        self.cost_map = self.map.copy()[:,:,0].astype('float64')
        for _ in range(self.con_size):
            tmp = self.cost_map.copy()
            tmp[1:] += self.cost_map[:-1] 
            tmp[:-1] += self.cost_map[1:] 
            tmp[:,1:] += self.cost_map[:,:-1] 
            tmp[:,:-1] += self.cost_map[:,1:]
            self.cost_map = np.clip(tmp, 0., 1.)
        for _ in range(self.con_size):
            tmp = self.cost_map.copy()
            tmp[1:] += np.sqrt(self.cost_map[:-1]) * 0.1
            tmp[:-1] += np.sqrt(self.cost_map[1:]) * 0.1
            tmp[:,1:] += np.sqrt(self.cost_map[:,:-1]) * 0.1
            tmp[:,:-1] += np.sqrt(self.cost_map[:,1:]) * 0.1
            self.cost_map = np.clip(tmp, 0., 1.)

    def _get_map(self):
        image = np.zeros([512,512,1]).astype(np.uint8)
        image[:, 0] = 255
        image[0, :] = 255
        image[:,-1] = 255
        image[-1,:] = 255
        for _ in range(15):
            x1 = np.random.randint(512)
            x2 = x1 + min(np.random.randint(50,100), 512-x1)
            y1 = np.random.randint(512)
            y2 = y1 + min(np.random.randint(50,100), 512-y1)
            image[x1:x2,y1:y2] = 255
        save_img = np.transpose(image.copy()[:,::-1], [1,0,2])
        fullpath = path.join(path.dirname(__file__), "map_{:03d}.png".format(self.rank+500))
        imageio.imwrite(fullpath, save_img)
        # image = imageio.imread(fullpath).reshape([512,512,1])
        return image

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
    def __init__(self, rank, **kwargs):
        super().__init__(rank, **kwargs)
        self.img_size = 52
        self.observation_space = Tuple((
            Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64),
            Box(low=-np.inf, high=np.inf, shape=(1,self.img_size,self.img_size), dtype=np.float64),
        ))
        # self.observation_space = Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)
        aspace_low = np.array([-0.6, -0.6, -0.6])
        aspace_high = np.array([0.6, 0.6, 0.6])
        self.action_space = Box(
            low=aspace_low, high=aspace_high, shape=(3,), dtype=np.float64
        )
        self.kp = np.array([50, 50, 50])
        self.kd = np.array([0.01, 0.01, 0.01])
        self.goal = np.zeros(2)
        self.t = 0.
        self.total_cnt = 0

    def reset(self):
        self.total_cnt += 1
        self.con_cnt = 0
        regenerate = True
        while regenerate:
            if np.random.rand() < 0.1:
                self._init()
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
        while True:
            self.goal = np.random.rand(2) * 9. - 4.5
            if self.total_cnt < 100:
                if np.random.rand() < 0.05:
                    vec = self.goal-np.array([pos_x,pos_y])
                    dist = np.linalg.norm(vec)
                    if dist > .1:
                        self.goal = vec/dist*0.1 + np.array([pos_x,pos_y])
            coor = self._pos2map(self.goal)
            if self.map[coor[0], coor[1], 0] == 0:
                break
        qpos = [pos_x, pos_y, yaw, 0.3]
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

        pos = self.sim.data.qpos.copy().flat[:2]
        coor = self._pos2map(pos)
        mx1 = coor[0]
        mx2 = coor[0]+self.obs_size*2
        my1 = coor[1]
        my2 = coor[1]+self.obs_size*2
        local_map = self.obs_map[:,mx1:mx2, my1:my2]
        local_map = local_map[:,0::2,0::2] + local_map[:,1::2,1::2] + local_map[:,0::2,1::2] + local_map[:,1::2,0::2]
        local_map = local_map[:,0::2,0::2] + local_map[:,1::2,1::2] + local_map[:,0::2,1::2] + local_map[:,1::2,0::2]
        local_map = (local_map!=0) * 1.

        return observation, local_map

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
        coor = self._pos2map(position)
        rew += (1-self.cost_map[coor[0],coor[1]]) * 0.01 * (3-self.con_cnt)
        self.prev_dist = dist.copy()
        return rew

    def _pos2map(self, pos):
        norm = pos / 10. + 0.5
        coor = (norm * 512).astype('long')
        coor = np.clip(coor, 0, 511)
        return coor

    def _get_done(self):

        done = self.t > 15.
        pos = self.sim.data.qpos.copy().flat[:2]
        coor = self._pos2map(pos)
        mx1 = coor[0]
        mx2 = coor[0]+self.con_size*2+1
        my1 = coor[1]
        my2 = coor[1]+self.con_size*2+1
        local_map = self.con_map[0, mx1:mx2, my1:my2]
        con_done = local_map.any()
        self.con_cnt += con_done
    
        return done or self.con_cnt>3

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

        