import numpy as np
from os import path
from gym import utils
from onpolicy.envs.mujoco.base_env import BaseEnv
from onpolicy.envs.mujoco.xml_gen import get_xml
from onpolicy.envs.mujoco.astar import find_path
from gym.spaces import Box, Tuple
import time

class NavigationEnv(BaseEnv):
    def __init__(self, num_agents, **kwargs):
        # hyper-para
        self.arrive_dist = 1.
        self.mlen = 500 # global map length (pixels)
        self.msize = 10. # map size (m)
        self.lmlen = 57 # local map length (pixels)
        self.warm_step = 4 # warm-up: let everything stable (s)
        self.frame_skip = 100 # 100/frame_skip = decision_freq (Hz)
        self.num_obs = 10
        self.hist_len = 4
        self.num_agent = num_agents
        self.domain_random_scale = 0. # TODO
        self.measure_random_scale = 5e-3
        self.init_kp = np.array([[2000, 2000, 800]])
        self.init_kd = np.array([[0.02, 0.02, 0.0]])
        self.max_axis_torque = 100.
        # self.init_ki = np.array([[0.0, 0.0, 10.]])
        # simulator
        load_mass = np.array([0., 1., 3., 5.])[self.num_agent]
        self.load_mass = load_mass * (1 + (np.random.rand()-.5) * self.domain_random_scale)
        self.cable_len = 1. * (1 + (np.random.random(self.num_agent)-.5) * self.domain_random_scale)
        self.anchor_id = np.random.randint(0, 4, self.num_agent)
        self.fric_coef = 1. * (1 + (np.random.random(self.num_agent)-.5) * self.domain_random_scale)
        model = get_xml(
            dog_num = self.num_agent, 
            obs_num = self.num_obs, 
            anchor_id = self.anchor_id,
            load_mass = self.load_mass,
            cable_len = self.cable_len,
            fric_coef = self.fric_coef
        )
        super().__init__(model, **kwargs)
        
        # observation space 
        self.hist_vec_obs_size = 13
        obs_size = self.hist_vec_obs_size # * self.hist_len
        self.observation_space = Tuple((
            Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64), 
            Box(low=-np.inf, high=np.inf, shape=(3,self.lmlen,self.lmlen), dtype=np.float64),
        ))
        # self.hist_vec_sta_size = 5 + 8*self.num_agent
        # sta_size = 5 + 5*self.num_agent + self.hist_vec_sta_size * (self.hist_len-1) + self.num_agent + 4*self.num_agent + 1
        sta_size = (obs_size + 9) * self.num_agent + 1 + 20 # + self.num_agent
        self.share_observation_space = Tuple((
            Box(low=-np.inf, high=np.inf, shape=(sta_size,), dtype=np.float64), 
            Box(low=-np.inf, high=np.inf, shape=(self.num_agent*3,self.lmlen,self.lmlen), dtype=np.float64),
        ))
        # action space
        aspace_low = np.array([-0.15, -0.05, -0.5])
        aspace_high = np.array([0.5, 0.05, 0.5])
        self.action_space = Box(
            low=aspace_low, high=aspace_high, shape=(3,), dtype=np.float64
        )
        # hyper-para
        bounds = self.model.actuator_ctrlrange.copy()
        self.torque_low, self.torque_high = bounds.astype(np.float32).T
        self.torque_low = self.torque_low.reshape([self.num_agent, self.action_space.shape[0]])
        self.torque_high = self.torque_high.reshape([self.num_agent, self.action_space.shape[0]])
        
        # wall
        self.wall_pos = np.array([
            [self.msize/2+1, 0.],
            [-self.msize/2-1, 0.],
            [0., self.msize/2+1],
            [0., -self.msize/2-1],
        ])
        self.wall_yaw = np.array([[np.pi/2],[np.pi/2],[0.],[0.]])
        
        self.arrive_time = 0.
        self.last_done = False
        self.total_step = 0

    def seed(self, seed):
        super().seed(seed)
        np.random.seed(seed)

    def reset(self):
        # init random tasks
        # np.random.seed(100)
        self.kp  = self.init_kp * (1. + (np.random.random(3)-.5) * self.domain_random_scale)
        self.kd  = self.init_kd * (1. + (np.random.random(3)-.5) * self.domain_random_scale)
        # self.ki  = self.init_ki * (1. + (np.random.random(3)-.5) * self.domain_random_scale)
        # init variables
        self.t = 0.
        self.max_time = 1e6
        self.last_cmd = np.zeros([self.num_agent, self.action_space.shape[0]])
        self.intergral = np.zeros([self.num_agent, self.action_space.shape[0]])
        self.hist_vec_obs = np.zeros([self.num_agent, self.hist_len, self.hist_vec_obs_size]) 
        self.hist_img_obs = np.zeros([self.num_agent, self.hist_len, *self.observation_space[1].shape])
        self.prev_output_vel = np.zeros([self.num_agent, self.action_space.shape[0]])
        # idx
        # regenerate env
        init_load_pos = (np.random.random(2)-.5) * self.msize 
        init_load_yaw = np.random.random(1) * 2 * np.pi 
        init_load_z = np.ones(1) * 0.55
        init_load = np.concatenate([init_load_pos, init_load_yaw, init_load_z], axis=-1).flatten()
        self.init_obs_pos = (np.random.random([self.num_obs, 2])-.5) * self.msize 
        self.init_obs_yaw = np.random.random([self.num_obs, 1]) * np.pi
        init_obs_z = np.ones([self.num_obs, 1]) * 0.55
        init_obs = np.concatenate([self.init_obs_pos, self.init_obs_yaw, init_obs_z], axis=-1).flatten()
        idx = 4+4*self.num_agent+4*self.num_obs
        init_wall = self.sim.data.qpos.copy()[idx:idx+12]
        init_dog_load_len = np.random.random([self.num_agent, 1]) * 0.25 + 0.75
        init_dog_load_yaw = init_load_yaw + (np.random.random([self.num_agent, 1])-.5) * np.pi
        init_dog_yaw = np.random.random([self.num_agent, 1]) * np.pi * 2
        init_dog_pos = self._get_toward(init_dog_load_yaw+init_load_yaw)[0] * init_dog_load_len
        anchor_id = self.anchor_id.reshape([self.num_agent, 1])
        init_dog_pos += self._get_toward(init_load_yaw)[0] * 0.3 * (anchor_id==0)
        init_dog_pos += self._get_toward(init_load_yaw)[1] * 0.3 * (anchor_id==1)
        init_dog_pos += self._get_toward(init_load_yaw)[0] * (-0.3) * (anchor_id==2)
        init_dog_pos += self._get_toward(init_load_yaw)[1] * (-0.3) * (anchor_id==3)
        init_dog_pos += init_load_pos.reshape([1,2])
        init_dog_z = np.ones([self.num_agent, 1]) * 0.3
        init_dog = np.concatenate([init_dog_pos, init_dog_yaw, init_dog_z], axis=-1).flatten()
        obs_dist = 0.
        if np.random.rand() < 0. and self.last_done:
            self.goal = self.sim.data.qpos.copy()[:2]
        else:
            while obs_dist < 0.8:
                self.goal = (np.random.random(2)-.5) * (self.msize-2.)
                obs_dist = np.linalg.norm(self.goal.reshape([1,2])-self.init_obs_pos, axis=-1).min()
        qpos = np.concatenate([init_load, init_dog, init_obs, init_wall, np.zeros([20])])
        self.set_state(np.array(qpos), np.zeros_like(qpos))
        for _ in range(self.warm_step):
            terminated, _ = self._do_simulation(self.last_cmd.copy(), self.frame_skip)
            # regenerate if done
            if terminated:
                return self.reset()
        # astar
        # draw obstacle map
        obs_len = np.ones([2])
        obs_map = self._draw_obs_map(
            self.init_obs_pos, self.init_obs_yaw, obs_len
        )
        # do astar
        load_pos = (init_load_pos / self.msize + .5) * self.mlen
        goal_pos = (self.goal / self.msize + .5) * self.mlen
        astar_path = find_path(load_pos, goal_pos, obs_map)
        
        if astar_path is None:
            return self.reset()
        
        self.path = []
        for i in range(9):
            idx = int(len(astar_path)/10*i)
            self.path.append(astar_path[idx][0])
            self.path.append(astar_path[idx][1])
        self.path.append(self.goal[0])
        self.path.append(self.goal[1])
        
        qpos = np.concatenate([qpos[:-20], self.path])
        self.set_state(np.array(qpos), np.zeros_like(qpos))
        
        # init variables
        self.t = 0.
        self.arrive_time = 0.
        self.total_rew = 0.
        load_pos = self.sim.data.qpos.copy()[:2]
        self.init_dist = np.linalg.norm(self.goal - load_pos)
        self.max_time = self.init_dist * 8. + 20.
        # self.obs_map = self._get_obs_map(self.init_obs_pos, self.init_obs_yaw)
        # RL_info
        observation = self._get_obs() 
        info = dict()
        # post process
        self._post_update(self.last_cmd)
        self.last_done = False
        return observation, info
    
    def _get_obs(self):
        
        cur_vec_obs, cur_img_obs, cur_vec_sta = self._get_cur_obs()
        self._update_history(cur_vec_obs, cur_img_obs)
        vec_obs = cur_vec_obs.reshape([self.num_agent, -1]) # self.hist_vec_obs.reshape([self.num_agent, -1])
        img_obs = cur_img_obs.reshape([self.num_agent, -1, *self.observation_space[1].shape[-2:]]) # self.hist_img_obs.reshape([self.num_agent, -1, *self.observation_space[1].shape[-2:]])
        vec_sta = cur_vec_sta.reshape([1, -1])
        # h_rew = self._get_h_rew()
        vec_sta = np.concatenate([vec_sta, [[self.max_time-self.t]]], -1)
        path = np.expand_dims(self.path, 0)
        vec_sta = np.concatenate([vec_sta, path], axis=-1)  
        vec_sta = np.repeat(vec_sta, self.num_agent, axis=0)
        # vec_idx = np.eye(self.num_agent)
        # vec_sta = np.concatenate([vec_sta, vec_idx], -1)
        img_sta = img_obs.copy().reshape([1, -1, *self.observation_space[1].shape[-2:]])
        img_sta = np.repeat(img_sta, self.num_agent, axis=0)
        # print("A", vec_obs.shape, img_obs.shape, vec_sta.shape, img_sta.shape)
        return vec_obs, img_obs, vec_sta, img_sta
    
    def _get_h_rew(self):
        self.gamma = 0.99
        state = self.sim.data.qpos.copy().flatten()
        dist = np.linalg.norm(state[:2]-self.goal)
        remain_t = self.max_time - self.t
        h_rew = self.gamma**(remain_t//self.dt) * dist
        return h_rew

    def _update_history(self, cur_vec_obs, cur_img_obs):
        
        self.hist_vec_obs[:,:-1] = self.hist_vec_obs[:,1:]
        self.hist_img_obs[:,:-1] = self.hist_img_obs[:,1:]
        self.hist_vec_obs[:,-1] = cur_vec_obs.copy()
        self.hist_img_obs[:,-1] = cur_img_obs.copy()

    def step(self, cmd):
        """
        observation(tuple): 
        reward(np.array):  [num_agent, 1]
        done(np.array):  [num_agent]
        info(dict): 
        """
        # pre process
        command = np.clip(cmd, self.action_space.low, self.action_space.high)
        # action = command.copy()
        done, contact = self._do_simulation(command, self.frame_skip)
        self.t += self.dt
        # update RL info
        observation = self._get_obs()
        reward, info = self._get_reward(contact)
        self.total_rew += reward
        # post process
        self._post_update(command)
        self.last_done |= (done and ~contact)
        return observation, reward, done, False, info

    def _get_done(self, dt): 

        terminate, contact = False, False
        # contact done
        for i in range(self.sim.data.ncon):
            con = self.sim.data.contact[i]
            obj1 = self.sim.model.geom_id2name(con.geom1)
            obj2 = self.sim.model.geom_id2name(con.geom2)
            if obj1=="floor" or obj2=="floor":
                continue
            if "obstacle" in obj1 or "obstacle" in obj2:
                terminate, contact = True, True
            else:
                terminate, contact = True, False
            return terminate, contact
        # contact rope done
        state = self.sim.data.qpos.copy().flatten()
        dog_state = state[4:4+self.num_agent*4]
        dog_state = dog_state.reshape([self.num_agent, 4])
        load_pos = state[:2].reshape([1, 2])
        load_dog_vec = dog_state[:,:2] - load_pos
        load_dog_yaw = np.arctan2(load_dog_vec[:,1], load_dog_vec[:,0])
        anchor_yaw = self.anchor_id * np.pi/2 + self.sim.data.qpos.copy()[2]
        yaw_cos = np.cos(load_dog_yaw-anchor_yaw)
        if (yaw_cos<-0.25).any():
            terminate, contact = True, False
            return terminate, contact
        # out of time
        if self.t > self.max_time:
            terminate, contact = True, False
            return terminate, contact
        # # arrive
        # dist = np.linalg.norm(self.sim.data.qpos[:2] - self.goal)
        # self.arrive_time = self.arrive_time + dt if dist < self.arrive_dist else 0.
        # if self.arrive_time >= self.arrive_dist:
        #     terminate, contact = True, False
        
        return terminate, contact

    def _get_reward(self, contact):

        rewards = []
        # pre-process
        weights = np.array([1., 2., 0.])
        weights = weights / weights.sum()
        state = self.sim.data.qpos.copy().flatten()
        # dist_per_step
        dist = np.linalg.norm(state[:2]-self.goal)
        rewards.append((self.prev_dist-dist)*(dist>self.arrive_dist))
        # goal_reach
        duration = self.frame_skip / 100
        max_speed = np.linalg.norm(self.action_space.high[:2])
        rewards.append((dist<=self.arrive_dist)*max_speed*duration)
        # contact
        rewards.append(contact)
        
        rew_dict = dict()
        rew_dict["dist_per_step"] = rewards[0]
        rew_dict["goal_reach"] = rewards[1]
        rew_dict["contact_done"] = rewards[2]
        rews = np.dot(rewards, weights)
        
        return rews, rew_dict
    
    def _cart2polar(self, coor):
        dist = np.linalg.norm(coor, axis=-1, keepdims=True)
        theta = np.arctan2(coor[:,1:], coor[:,:1])
        cos, sin = np.cos(theta), np.sin(theta)
        polar = np.concatenate([dist, cos, sin], axis=-1)
        return polar

    def _get_cur_obs(self):

        # vector: partial observation
        load_pos = self.sim.data.qpos.copy()[0:2].reshape([1,2])
        load_position = (self.goal.reshape([1,2])-load_pos) / self.msize
        load_position = self._cart2polar(load_position)
        load_position = np.repeat(load_position, self.num_agent, axis=0)
        load_yaw = self.sim.data.qpos.copy()[2:3].reshape([1,1])
        load_yaw = np.repeat(load_yaw, self.num_agent, axis=0)
        load_yaw = [np.sin(load_yaw), np.cos(load_yaw)]
        dog_state = self.sim.data.qpos.copy()[4:4+4*self.num_agent]
        dog_state = dog_state.reshape([self.num_agent, 4])
        dog_pos = (self.goal.reshape([1,2])-dog_state[:,0:2]) / self.msize
        dog_position = self._cart2polar(dog_pos)
        dog_yaw = dog_state[:,2:3]
        dog_yaw = [np.sin(dog_yaw), np.cos(dog_yaw)]
        anchor_yaw = self.anchor_id * np.pi/2 + self.sim.data.qpos.copy()[2]
        anchor_vec, _ = self._get_toward(anchor_yaw.reshape([self.num_agent, 1]))
        anchor_vec = self._cart2polar(anchor_vec)
        cur_vec_obs = [load_position, dog_position] + load_yaw + dog_yaw + [anchor_vec] # TODO
        cur_vec_obs = np.concatenate(cur_vec_obs, axis=-1) 
        cur_vec_obs += (np.random.random(cur_vec_obs.shape)-.5) * self.measure_random_scale
        # print("cur_vec_obs.shape", cur_vec_obs.shape) # [num_agent, vec_shape]
        cur_vec_sta = [self.error, self.d_output, self.prev_output_vel]
        cur_vec_sta = np.concatenate(cur_vec_sta, axis=-1) 
        cur_vec_sta = np.concatenate([cur_vec_obs, cur_vec_sta], axis=-1)
        # image: partial observation
        cur_img_obs = []
        for i in range(self.num_agent):
            cur_img_o = []
            dog_pos = self.sim.data.qpos.copy()[4+4*i:4+4*i+2]
            dog_yaw = self.sim.data.qpos.copy()[4+4*i+2]
            obs_len = np.ones([2])
            obs_map = self._draw_map(
                dog_pos, dog_yaw, 
                self.init_obs_pos, self.init_obs_yaw, obs_len
            )
            wall_len = np.array([self.msize+2, 2.])
            obs_map += self._draw_map(
                dog_pos, dog_yaw, 
                self.wall_pos, self.wall_yaw, wall_len
            )
            cur_img_o.append((obs_map!=0)*1.)
            
            # obs_map = np.zeros(self.observation_space[1].shape[1:]) # TODO
            # cur_img_o.append((obs_map!=0)*1.)
            
            load_pos = self.sim.data.qpos.copy()[0:2].reshape([1,2])
            load_yaw = self.sim.data.qpos.copy()[2].reshape([1,1])
            load_len = np.ones([2]) * 0.6
            load_map = self._draw_map(
                dog_pos, dog_yaw, 
                load_pos, load_yaw, load_len
            )
            cur_img_o.append((load_map!=0)*1.)
            
            dog_pos = self.sim.data.qpos.copy()[4+4*i:4+4*i+2]
            dog_yaw = self.sim.data.qpos.copy()[4+4*i+2]
            all_dog_state = self.sim.data.qpos.copy()[4:4+4*self.num_agent]
            all_dog_state = all_dog_state.reshape([self.num_agent, 4])
            all_dog_state = np.delete(all_dog_state, i, axis=0)
            all_dog_pos = all_dog_state[:,0:2]
            all_dog_yaw = all_dog_state[:,2:3]
            dog_len = np.array([0.65, 0.3])
            dog_map = self._draw_map(
                dog_pos, dog_yaw, 
                all_dog_pos, all_dog_yaw, dog_len
            )
            cur_img_o.append((dog_map!=0)*1.)
            
            cur_img_o = np.stack(cur_img_o, axis=0)
            cur_img_obs.append(cur_img_o)
        cur_img_obs = np.stack(cur_img_obs, axis=0)
        # print("cur_img_obs.shape", cur_img_obs.shape) # [num_agent, vec_shape]
        # import imageio
        # imageio.imwrite("../envs/mujoco/assets/dog.png", dog_map)
        return cur_vec_obs, cur_img_obs, cur_vec_sta
    
    def _draw_obs_map(self, box_pos, box_yaw, box_len):
        """ draw dog/box/obstacle local map

        Args:
            dog_pos (np.array): size = [2]
            dog_theta (float): 
            box_pos (np.array): size = [box_num, 2]
            box_yaw (np.array): size = [box_num, 1]
            box_len (np.array): size = [2]
        """
        box_len /= 2
        
        box_pos = box_pos
        box_tow, box_ver = self._get_toward(box_yaw)
        box_rect = self._get_rect(box_pos, box_tow, box_ver, box_len)
        box_map = np.zeros([self.mlen, self.mlen])
        min_idx = -self.lmlen*3//2
        max_idx = self.mlen + self.lmlen*3//2 + 1
        for rect in box_rect:
            norm_rect = (rect.copy() / self.msize + .5) * self.mlen
            norm_rect = norm_rect.astype('long')
            tmp_map = self._draw_rect(norm_rect, min_idx, max_idx)
            box_map += tmp_map[-min_idx:self.mlen-min_idx,-min_idx:self.mlen-min_idx]
        box_map = (box_map!=0) * 1.
        
        return box_map

    def _draw_map(self, dog_pos, dog_theta, box_pos, box_yaw, box_len):
        """ draw dog/box/obstacle local map

        Args:
            dog_pos (np.array): size = [2]
            dog_theta (float): 
            box_pos (np.array): size = [box_num, 2]
            box_yaw (np.array): size = [box_num, 1]
            box_len (np.array): size = [2]
        """
        box_len /= 2
        
        box_pos = box_pos - dog_pos.reshape([1,2])
        box_tow, box_ver = self._get_toward(box_yaw)
        box_rect = self._get_rect(box_pos, box_tow, box_ver, box_len)
        rotate_mat = np.array([[
            [np.cos(dog_theta), -np.sin(dog_theta)], 
            [np.sin(dog_theta), np.cos(dog_theta)]
        ]])
        box_rect = box_rect @ rotate_mat
        box_map = np.zeros([self.lmlen, self.lmlen])
        min_idx = self.mlen//2 - self.lmlen*3//2
        max_idx = self.mlen//2 + self.lmlen*3//2 + 1
        for rect in box_rect:
            norm_rect = (rect.copy() / self.msize + .5) * self.mlen
            norm_rect = norm_rect.astype('long')
            m = self._draw_rect(norm_rect, min_idx, max_idx)
            box_map += \
                m[0::3,0::3] + m[1::3,0::3] + m[2::3,0::3] + \
                m[0::3,1::3] + m[1::3,1::3] + m[2::3,1::3] + \
                m[0::3,2::3] + m[1::3,2::3] + m[2::3,2::3] 
        box_map = (box_map!=0) * 1.
        
        return box_map
        
        
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

    def _draw_rect(self, rect, min_idx, max_idx):
        
        # get line function y = ax + b
        lines = [[rect[i], rect[(i+1)%4]] for i in range(4)]
        lines = np.array(lines)
        a = (lines[:,0,1] - lines[:,1,1]) / (lines[:,0,0] - lines[:,1,0])
        b = lines[:,0,1] - a * lines[:,0,0]
        # get x,y coor idx
        length = max_idx - min_idx
        x_axis = np.arange(length) + .5 + min_idx #[0]
        y_axis = np.arange(length) + .5 + min_idx #[1]
        y_map, x_map = np.meshgrid(y_axis, x_axis)
        if (a > 10).any():
            x_min = np.min(lines[:,:,0])
            x_max = np.max(lines[:,:,0])
            y_min = np.min(lines[:,:,1])
            y_max = np.max(lines[:,:,1])
            b_map = (x_map>x_min) & (x_map<x_max) \
                & (y_map>y_min) & (y_map<y_max)
            return b_map.astype('long')
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

    def _do_simulation(self, action, n_frames):
        """ 
        (input) action: the desired velocity of the agent. shape=(num_agent, action_size)
        (input) n_frames: the agent will use this action for n_frames*dt seconds
        """
        # print("B", action)
        for _ in range(n_frames):
            # assert np.array(ctrl).shape==self.action_space.shape, "Action dimension mismatch"
            state = self.sim.data.qvel.flat.copy()[4:4+self.num_agent*4]
            state = state.reshape([self.num_agent, 4])
            cur_vel = self._global2local(state[:,:3])
            local_action = self._pd_contorl(target_vel=action, cur_vel=cur_vel, dt=self.dt/n_frames)
            self.sim.data.ctrl[:] = self._local2global(local_action).flatten()
            self.sim.step()
            terminate, contact = self._get_done(self.dt/n_frames)
            if terminate:
                break
        return terminate, contact

    def _pd_contorl(self, target_vel, cur_vel, dt):
        """
        given the desired velocity, calculate the torque of the actuator.
        (input) target_vel: [num_agent, 3]
        (input) cur_vel: [num_agent, 3]
        (input) dt: float 
        (output) torque
        """
        self.error = target_vel - cur_vel
        # self.intergral += error * self.ki * dt
        # self.intergral = np.clip(self.intergral, self.torque_low, self.torque_high)
        self.d_output = cur_vel - self.prev_output_vel
        torque = self.kp * self.error - self.kd * (self.d_output/dt) # + self.intergral
        torque_low = np.repeat(np.array([[-13*9.83,10.,-100]]), self.num_agent, 0)
        torque_high = np.repeat(np.array([[13*9.83,10.,100]]), self.num_agent, 0)
        torque = np.clip(torque, torque_low, torque_high)
        self.prev_output_vel = cur_vel.copy()
        return torque.flatten()

    def _post_update(self, cmd):
        """
        restore the history information
        """
        # update last distance 
        load_pos = self.sim.data.qpos.copy()[0:2]
        self.prev_dist = np.linalg.norm(load_pos-self.goal)
        self.last_load_pos = load_pos.copy()
        self.total_step += 1

    def _local2global(self, input_action):

        state = self.sim.data.qpos.flat.copy()[4:4+self.num_agent*4]
        state = state.reshape([self.num_agent,4])
        tow, ver = self._get_toward(state[:,2:3])
        input_action = input_action.reshape([self.num_agent,3])
        global_action = tow * input_action[:,0:1] + ver * input_action[:,1:2]
        output_action = np.concatenate([global_action, input_action[:,2:].copy()], axis=-1)
        return output_action
    
    def _global2local(self, input_action):
        
        state = self.sim.data.qpos.flat.copy()[4:4+self.num_agent*4]
        theta = state.reshape([self.num_agent,4])[:,2]
        rot_mat = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        rot_mat = np.transpose(rot_mat, [2,0,1])
        action = np.expand_dims(input_action[:,:2], 2)
        output_action = rot_mat @ action
        output_action = output_action.squeeze(2) 
        output_action = np.concatenate([output_action, input_action[:,2:3]], -1)
        
        return output_action
    
    
    
    
    


