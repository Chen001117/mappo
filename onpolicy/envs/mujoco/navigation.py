import numpy as np
from os import path
from gym import utils
from onpolicy.envs.mujoco.base_env import BaseEnv
from onpolicy.envs.mujoco.xml_gen import get_xml
from gym.spaces import Box, Tuple

class NavigationEnv(BaseEnv):
    def __init__(self, **kwargs):
        # hyper-para
        self.max_time = 15
        self.lmlen = 29 # local map length (pixels)
        assert self.lmlen%2==1, "The length of the local map should be an odd number"
        self.warm_step = 10 # 
        self.msize = 10. # map size (m)
        self.mlen = 400 # global map length (pixels)
        self.random_scale = 0.01 # add noise to the observed coordinate (m)
        self.frame_skip = 4 
        self.num_agent = 1
        self.num_obs = 15
        self.hist_len = 4
        self.kp = np.array([[50, 50, 50]])
        self.kd = np.array([[0.01, 0.01, 0.01]])
        # simulator
        model = get_xml(dog_num=self.num_agent, obs_num=self.num_obs)
        super().__init__(model, **kwargs)
        # observation space 
        self.observation_space = Tuple((
            Box(low=-np.inf, high=np.inf, shape=(35,), dtype=np.float64),
            Box(low=-np.inf, high=np.inf, shape=(1,self.lmlen,self.lmlen), dtype=np.float64),
        ))
        # action space
        aspace_low = np.array([-0.6, -0.6, -0.6])
        aspace_high = np.array([0.6, 0.6, 0.6])
        self.action_space = Box(
            low=aspace_low, high=aspace_high, shape=(3,), dtype=np.float64
        )
        # hyper-para
        bounds = self.model.actuator_ctrlrange.copy()
        self.torque_low, self.torque_high = bounds.astype(np.float32).T

    def reset(self):
        # init variables
        self.last_cmd = np.zeros([self.num_agent, self.action_space.shape[0]])
        hist_size = self.action_space.shape[0] + 4 
        self.hist_obs = np.zeros([self.hist_len, self.num_agent, hist_size])
        # init state 
        regenerate = True
        while regenerate:
            self.prev_output = np.zeros([self.num_agent, self.action_space.shape[0]])
            init_obs_pos = (np.random.random([self.num_obs, 2])-.5) * self.msize 
            init_obs_yaw = np.random.random([self.num_obs, 1]) * np.pi
            init_obs_z = np.ones([self.num_obs, 1]) * 0.55
            init_pos = (np.random.random([self.num_agent, 2])-.5) * self.msize
            init_yaw = np.random.random([self.num_agent, 1]) * np.pi * 2.
            init_z = np.ones([self.num_agent, 1]) * 0.2
            self.goal = (np.random.random([self.num_agent, 2])-.5) * self.msize 
            dog_p = np.concatenate([init_pos, init_yaw, init_z], axis=-1).flatten()
            obs_p = np.concatenate([init_obs_pos, init_obs_yaw, init_obs_z], axis=-1).flatten()
            qpos = np.concatenate([dog_p, obs_p, self.goal.flatten()])
            self.set_state(np.array(qpos), np.zeros_like(qpos))
            for _ in range(self.warm_step):
                self._do_simulation(self.last_cmd.copy(), self.frame_skip)
            self.t = 0.
            regenerate = self._get_done()
        self.obs_map = self._get_obs_map(init_obs_pos, init_obs_yaw)
        # update RL_info
        obs = self._get_obs()
        info = dict()
        # post process
        self._post_update(self.last_cmd.copy())
        return obs, info

    def step(self, command):
        # pre process
        command = np.clip(command, self.action_space.low, self.action_space.high)
        action = self._local_to_global(command)
        self._do_simulation(action, self.frame_skip)
        self.t += self.dt
        # update RL info
        observation = self._get_obs()
        terminated = self._get_done()
        reward = self._get_reward()
        info = dict()
        # post process
        self._post_update(command)
        return observation, reward, terminated, False, info

    def _get_toward(self, theta):
        # theta : [env_num, 1]

        forward = np.zeros([*theta.shape[:-1], 1, 2])
        vertical = np.zeros([*theta.shape[:-1], 1, 2])
        forward[...,0] = 1
        vertical[...,1] = 1
        cos = np.cos(theta)
        sin = np.sin(theta)
        rotate = np.concatenate([cos, sin, -sin, cos], -1)
        rotate = rotate.reshape([*theta.shape[:-1], 2, 2])
        forward = forward @ rotate
        vertical = vertical @ rotate
        forward = forward.squeeze(-2)
        vertical = vertical.squeeze(-2)

        return forward, vertical
    
    def _get_rect(self, cen, forw, vert, length):
        '''
        cen(input): [*shape, 2] 
        forw(input): [*shape, 2]
        vert(input): [*shape, 2]
        length(input): [*shape, 2]
        vertice(output): [*shape, 4, 2]
        '''

        vertice = np.stack([
            cen + forw * length[...,0:1] + vert * length[...,1:2],
            cen - forw * length[...,0:1] + vert * length[...,1:2],
            cen - forw * length[...,0:1] - vert * length[...,1:2],
            cen + forw * length[...,0:1] - vert * length[...,1:2],
        ], axis=-2)
        
        return vertice

    def draw_rect(self, rect):
        rect = (rect / self.msize + .5) * self.mlen
        # get line function y = ax + b
        lines = [[rect[i], rect[(i+1)%4]] for i in range(4)]
        lines = np.array(lines)
        a = (lines[:,0,1] - lines[:,1,1]) / (lines[:,0,0] - lines[:,1,0])
        b = lines[:,0,1] - a * lines[:,0,0]
        # get x,y coor idx
        x_axis = np.arange(self.mlen) + .5
        y_axis = np.arange(self.mlen) + .5
        y_map, x_map = np.meshgrid(x_axis, y_axis)
        x_map = np.expand_dims(x_map, 0) 
        x_map = np.repeat(x_map, 4, 0)
        y_map = np.expand_dims(y_map, 0) 
        y_map = np.repeat(y_map, 4, 0)
        b_map = np.zeros_like(y_map).astype('bool')
        # fill rect 
        for i in range(4):
            b_map[i] = y_map[i] > a[i] * x_map[i] + b[i]
        b_map[0] = b_map[0] ^ b_map[2]
        b_map[1] = b_map[1] ^ b_map[3]
        b_map[0] = b_map[0] & b_map[1]
        return b_map[0].astype('long')

    def _get_obs_map(self, obs_pos, obs_yaw):
        obs_toward, obs_vert = self._get_toward(obs_yaw)
        obs_vertice = self._get_rect(obs_pos, obs_toward, obs_vert, np.array([0.6,0.6]))
        hmlen = (self.lmlen*3-1) // 2
        obs_map = np.zeros([1, self.mlen+hmlen*2, self.mlen+hmlen*2])
        for rect in obs_vertice:
            obs_map[0,hmlen:-hmlen,hmlen:-hmlen] += self.draw_rect(rect)
        obs_map = (obs_map!=0) * 1.
        return obs_map 

    def _do_simulation(self, action, n_frames):
        """ 
        (input) action: the desired velocity of the agent. shape=(num_agent, action_size)
        (input) n_frames: the agent will use this action for n_frames*dt seconds
        """
        for _ in range(n_frames):
            ctrl = self._pd_contorl(target_vel=action, dt=self.dt/n_frames).flatten()
            assert np.array(ctrl).shape==self.action_space.shape, "Action dimension mismatch"
            self.sim.data.ctrl[:] = ctrl
            self.sim.step()

    def _pd_contorl(self, target_vel, dt):
        """
        given the desired velocity, calculate the torque of the actuator.
        (input) target_vel: 
        (input) dt: 
        (output) torque
        """
        state = self.sim.data.qvel.copy()[:self.num_agent*4]
        cur_vel = state.reshape([self.num_agent, 4])[:,:3]
        error = target_vel - cur_vel
        d_output = cur_vel - self.prev_output
        torque = self.kp * error - self.kd * (d_output/dt)
        torque = np.clip(torque, self.torque_low, self.torque_high)
        self.prev_output = torque.copy()
        return torque

    def _post_update(self, cmd):
        """
        restore the history information
        """
        state = self.sim.data.qpos.copy()[:self.num_agent*4]
        state = state.reshape([self.num_agent, 4])
        pos, yaw = state[:,:2], state[:,2:3]
        sin_yaw, cos_yaw = np.sin(yaw), np.cos(yaw)
        self.hist_obs[:-1] = self.hist_obs[1:]
        cmd = cmd.reshape([self.num_agent, -1])
        self.hist_obs[-1] = np.concatenate([pos, sin_yaw, cos_yaw, cmd], axis=-1)
        self.prev_dist = np.linalg.norm(pos-self.goal)

    def _get_obs(self):
        
        # vec_obs
        state = self.sim.data.qpos.copy()[:self.num_agent*4]
        state = state.reshape([self.num_agent, 4])
        pos, yaw = state[:,:2], state[:,2:3]
        sin_yaw, cos_yaw = np.sin(yaw), np.cos(yaw)
        time = np.array([[self.t]]).repeat(self.num_agent,1)
        cur_obs = np.concatenate([pos, sin_yaw, cos_yaw, self.goal, time], axis=-1)
        hist_obs = self.hist_obs.transpose(1,0,2).reshape([self.num_agent, -1])
        observation = np.concatenate([cur_obs, hist_obs], axis=-1)
        # add random
        observation += np.random.random(observation.shape) * self.random_scale
        # img_obs
        coor = ((pos/self.msize+.5)*self.mlen).astype('long')
        hlmlen = (self.lmlen*3-1)//2
        x1, x2 = coor[:,0], coor[:,0]+hlmlen*2+1
        y1, y2 = coor[:,1], coor[:,1]+hlmlen*2+1
        local_map = []
        for i in range(self.num_agent):
            m = self.obs_map[:, x1[i]:x2[i], y1[i]:y2[i]]
            m = m[:,0::3,0::3] + m[:,1::3,0::3] + m[:,2::3,0::3] + \
                m[:,0::3,1::3] + m[:,1::3,1::3] + m[:,2::3,1::3] + \
                m[:,0::3,2::3] + m[:,1::3,2::3] + m[:,2::3,2::3] 
            local_map.append((m!=0)*1.)
        local_map = np.array(local_map)
        return observation, local_map

    def _local_to_global(self, input_action):
        input_action = input_action.reshape([self.num_agent,3])
        local_action = input_action[:,:2].copy()
        local_action = local_action.reshape([self.num_agent,1,2])
        state = self.sim.data.qpos.flat.copy()[:self.num_agent*3]
        state = state.reshape([self.num_agent,3])
        theta = state[:,2] 
        rotate_mat = [
            np.cos(theta), np.sin(theta), -np.sin(theta), np.cos(theta),
        ]
        rotate_mat = np.stack(rotate_mat, axis=0)
        rotate_mat = rotate_mat.reshape([self.num_agent,2,2])
        global_action = local_action @ rotate_mat
        global_action = global_action.reshape([self.num_agent,2])
        output_action = np.concatenate([global_action, input_action[:,2:].copy()], axis=-1)
        return output_action

    def _get_done(self): 
        if self.t > self.max_time:
            return True
        state = self.sim.data.qpos.copy()[:self.num_agent*4]
        state = state.reshape([self.num_agent, 4])
        pos, yaw = state[:,:2], state[:,2:3]  
        if (abs(pos)>4.9).any():
            return True
        dist = np.linalg.norm(state[:,:2]-self.goal)
        if dist < 0.25:
            return True
        for i in range(self.sim.data.ncon):
            con = self.sim.data.contact[i]
            obj1 = self.sim.model.geom_id2name(con.geom1)
            obj2 = self.sim.model.geom_id2name(con.geom2)
            if obj1=="floor" or obj2=="floor":
                continue
            return True
        return False

    def _get_reward(self):
        state = self.sim.data.qpos.copy().flatten()[:self.num_agent*4]
        state = state.reshape([self.num_agent, 4])
        dist = np.linalg.norm(state[:,:2]-self.goal)
        rew = self.prev_dist - dist 
        dt = (15.-self.t) / self.dt
        rew = 0.05*(1-0.99**dt)/(1-0.99) if dist < 0.25 else rew 
        return rew


        