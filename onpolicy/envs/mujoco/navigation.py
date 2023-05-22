import numpy as np
from os import path
from gym import utils
from onpolicy.envs.mujoco.base_env import BaseEnv
from onpolicy.envs.mujoco.xml_gen import get_xml
from gym.spaces import Box, Tuple
import time

class NavigationEnv(BaseEnv):
    def __init__(self, num_agents, **kwargs):
        # hyper-para
        self.mlen = 500 # global map length (pixels)
        self.msize = 10. # map size (m)
        self.lmlen = 57 # local map length (pixels)
        self.warm_step = 4 # warm-up: let everything stable (s)
        self.frame_skip = 25 # 100/frame_skip = decision_freq (Hz)
        self.num_obs = 10
        self.hist_len = 4
        self.num_agent = num_agents
        self.domain_random_scale = 0.2
        self.measure_random_scale = 0.01
        self.init_kp = np.array([[2000, 2000, 80]])
        self.init_kd = np.array([[0.02, 0.02, 0.01]])
        self.init_ki = np.array([[0.0, 0.0, 10.]])
        # simulator
        self.load_mass = 5. * (1 + (np.random.rand()-.5) * self.domain_random_scale)
        self.cable_len = 1. * (1 + (np.random.random(self.num_agent)-.5) * self.domain_random_scale)
        self.anchor_id = np.random.randint(0, 4, self.num_agent)
        self.fric_coef = 1. * (1 + (np.random.random(self.num_agent)-.5) * self.domain_random_scale)
        self.max_torque = 9.83 * (13. + 2.)
        model = get_xml(
            dog_num = self.num_agent, 
            obs_num = self.num_obs, 
            anchor_id = self.anchor_id,
            load_mass = self.load_mass,
            cable_len = self.cable_len,
            fric_coef = self.fric_coef,
        )
        super().__init__(model, **kwargs)
        # observation space 
        self.hist_vec_obs_size = 13
        obs_size = 10 + self.hist_vec_obs_size * (self.hist_len-1) + 4
        self.observation_space = Tuple((
            Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64), 
            Box(low=-np.inf, high=np.inf, shape=(8,self.lmlen,self.lmlen), dtype=np.float64),
        ))
        self.hist_vec_sta_size = 5 + 8*self.num_agent
        sta_size = 5 + 5*self.num_agent + self.hist_vec_sta_size * (self.hist_len-1) + self.num_agent + 4*self.num_agent + 1
        self.share_observation_space = Tuple((
            Box(low=-np.inf, high=np.inf, shape=(sta_size,), dtype=np.float64), 
            Box(low=-np.inf, high=np.inf, shape=(self.num_agent*8+8,self.lmlen,self.lmlen), dtype=np.float64),
        ))
        # action space
        aspace_low = np.array([-0.25, -0.05, -0.5])
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

    def seed(self, seed):
        super().seed(seed)
        np.random.seed(seed)

    def reset(self):
        # init random tasks
        # np.random.seed(100)
        self.kp  = self.init_kp * (1. + (np.random.random(3)-.5) * self.domain_random_scale)
        self.ki  = self.init_ki * (1. + (np.random.random(3)-.5) * self.domain_random_scale)
        self.kd  = self.init_kd * (1. + (np.random.random(3)-.5) * self.domain_random_scale)
        # init variables
        self.t = 0.
        self.max_time = 1e6
        self.last_cmd = np.zeros([self.num_agent, self.action_space.shape[0]])
        self.intergral = np.zeros([self.num_agent, self.action_space.shape[0]])
        self.hist_vec_obs = np.zeros([self.num_agent, self.hist_len-1, self.hist_vec_obs_size]) 
        self.hist_img_obs = np.zeros([self.num_agent, self.hist_len-1, 2, *self.observation_space[1].shape[-2:]])
        self.hist_vec_sta = np.zeros([self.num_agent, self.hist_len-1, self.hist_vec_sta_size]) 
        self.hist_img_sta = np.zeros([self.num_agent, self.hist_len-1, 2+self.num_agent*2, *self.observation_space[1].shape[-2:]])
        self.prev_output_vel = np.zeros([self.num_agent, self.action_space.shape[0]])
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
        init_dog_pos = self._get_toward(init_dog_load_yaw)[0] * init_dog_load_len
        anchor_id = self.anchor_id.reshape([self.num_agent, 1])
        init_dog_pos += self._get_toward(init_load_yaw)[0] * 0.3 * (anchor_id==0)
        init_dog_pos += self._get_toward(init_load_yaw)[1] * 0.3 * (anchor_id==1)
        init_dog_pos += self._get_toward(init_load_yaw)[0] * (-0.3) * (anchor_id==2)
        init_dog_pos += self._get_toward(init_load_yaw)[1] * (-0.3) * (anchor_id==3)
        init_dog_pos += init_load_pos.reshape([1,2])
        init_dog_z = np.ones([self.num_agent, 1]) * 0.3
        init_dog = np.concatenate([init_dog_pos, init_dog_yaw, init_dog_z], axis=-1).flatten()
        obs_dist = 0.
        while obs_dist < 0.8:
            self.goal = (np.random.random(2)-.5) * (self.msize-2.)
            obs_dist = np.linalg.norm(self.goal.reshape([1,2])-self.init_obs_pos, axis=-1).min()
        qpos = np.concatenate([init_load, init_dog, init_obs, init_wall, self.goal.flatten()])
        self.set_state(np.array(qpos), np.zeros_like(qpos))
        for _ in range(self.warm_step):
            terminated, _ = self._do_simulation(self.last_cmd.copy(), self.frame_skip)
            # regenerate if done
            if terminated:
                return self.reset()
        # init variables
        self.t = 0.
        self.arrive_time = 0.
        load_pos = self.sim.data.qpos.copy()[:2]
        dist = np.linalg.norm(load_pos-self.goal, axis=-1)
        self.max_time = dist * 7.5 + 15.
        # self.obs_map = self._get_obs_map(self.init_obs_pos, self.init_obs_yaw)
        # RL_info
        cur_obs, observation = self._get_obs() 
        info = dict()
        # post process
        self._post_update(self.last_cmd, cur_obs)
        return observation, info
    
    def _get_obs(self):

        cur_obs = self._get_cur_obs()
        cur_vec_obs, cur_img_obs, cur_vec_sta, cur_img_sta = cur_obs
        anchor_id = self.anchor_id.reshape([self.num_agent, 1])
        anchor_vec = np.eye(4)[anchor_id][:,0]
        hist_vec_obs = self.hist_vec_obs.reshape([self.num_agent, -1])
        vec_obs = np.concatenate([cur_vec_obs, hist_vec_obs, anchor_vec], -1)
        hist_img_obs = self.hist_img_obs.reshape([self.num_agent, -1, *self.hist_img_obs.shape[-2:]])
        img_obs = np.concatenate([cur_img_obs, hist_img_obs], 1) 
        vec_sta_id = np.eye(self.num_agent)
        anchor_vec = np.eye(4)[anchor_id].reshape([1, self.num_agent*4])
        anchor_vec = np.repeat(anchor_vec, self.num_agent, axis=0)
        hist_vec_obs = self.hist_vec_obs.reshape([self.num_agent, -1])
        hist_vec_sta = self.hist_vec_sta.reshape([self.num_agent, -1])
        times = np.array([[self.max_time-self.t]])
        times = np.repeat(times, self.num_agent, axis=0)
        vec_sta = np.concatenate([cur_vec_sta, hist_vec_sta, anchor_vec, vec_sta_id, times], -1)
        hist_img_sta = self.hist_img_sta.reshape([self.num_agent, -1, *self.hist_img_sta.shape[-2:]])
        img_sta = np.concatenate([cur_img_sta, hist_img_sta], 1)
        obs = vec_obs, img_obs, vec_sta, img_sta

        return cur_obs, obs

    def step(self, cmd):
        """
        observation(tuple): 
        reward(np.array):  [num_agent, 1]
        done(np.array):  [num_agent]
        info(dict): 
        """
        # pre process
        command = np.clip(cmd, self.action_space.low, self.action_space.high)
        action = self._local_to_global(command)
        # action = command.copy()
        done, contact = self._do_simulation(action, self.frame_skip)
        self.t += self.dt
        # update RL info
        cur_obs, observation = self._get_obs()
        reward, info = self._get_reward(contact)
        # post process
        self._post_update(command, cur_obs)
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
            terminate, contact = True, True
            return terminate, contact
        # out of time
        if self.t > self.max_time:
            terminate, contact = True, False
            return terminate, contact
        # arrive
        dist = np.linalg.norm(self.sim.data.qpos[:2] - self.goal)
        self.arrive_time = self.arrive_time + dt if dist < 0.5 else 0.
        if self.arrive_time >= 3.:
            terminate, contact = True, False
        
        return terminate, contact

    def _get_reward(self, contact):

        rewards = []
        # pre-process
        weights = np.array([1., 2., 0.1])
        weights = weights / weights.sum()
        state = self.sim.data.qpos.copy().flatten()
        # goal_distance rewards
        dist = np.linalg.norm(state[:2]-self.goal)
        rewards.append((self.prev_dist-dist)*(dist>0.5))
        # goal reach rewards
        max_speed = np.linalg.norm(self.action_space.high[:2])
        duration = self.frame_skip / 100
        rew = (dist<0.5) * max_speed * duration
        if self.arrive_time >= 3.:
            gamma = 0.99
            remain_t = (self.max_time - self.t) // self.dt
            rew = rew * (1-gamma**remain_t) / (1-gamma)
        rewards.append(rew)
        # done penalty
        rewards.append(-contact*1.)
        
        # # colaborate rew
        # load_pos = state[:2]
        # load_move_vec = load_pos - self.last_load_pos
        # dog_state = state[4:4+self.num_agent*4]
        # dog_pos = dog_state.reshape([self.num_agent, 4])[:,:2]
        # dog_load_vecs = dog_pos - load_pos.reshape([1, 2])
        # cos_thetas = []
        # for dog_load_vec in dog_load_vecs:
        #     inner_dot = np.dot(dog_load_vec, load_move_vec)
        #     dist_load = np.linalg.norm(load_move_vec)
        #     dist_dog = np.linalg.norm(dog_load_vec)
        #     if dist_dog>1e-3 and dist_load>1e-3:
        #         cos_thetas.append(inner_dot/dist_load/dist_dog)
        #     else:
        #         cos_thetas.append(0.)
        # theta_weights = (np.array(cos_thetas)+1.)/2.

        rew_dict = dict()
        rew_dict["goal_distance"] = rewards[0]
        rew_dict["goal_reach"] = rewards[1]
        rew_dict["con_penalty"] = rewards[2]
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
        load_pos = (load_pos-self.goal.reshape([1,2])) / self.msize
        load_pos = self._cart2polar(load_pos)
        load_pos = np.repeat(load_pos, self.num_agent, axis=0)
        load_yaw = self.sim.data.qpos.copy()[2:3].reshape([1,1])
        load_yaw = np.repeat(load_yaw, self.num_agent, axis=0)
        load_yaw = [np.sin(load_yaw), np.cos(load_yaw)]
        dog_state = self.sim.data.qpos.copy()[4:4+4*self.num_agent]
        dog_state = dog_state.reshape([self.num_agent, 4])
        dog_pos = (dog_state[:,0:2]-self.goal.reshape([1,2])) / self.msize
        dog_pos = self._cart2polar(dog_pos)
        dog_yaw = dog_state[:,2:3]
        dog_yaw = [np.sin(dog_yaw), np.cos(dog_yaw)]
        cur_vec_obs = [load_pos, dog_pos] + load_yaw + dog_yaw
        cur_vec_obs = np.concatenate(cur_vec_obs, axis=-1) 
        cur_vec_obs += (np.random.random(cur_vec_obs.shape)-.5) * self.measure_random_scale
        # print("cur_vec_obs.shape", cur_vec_obs.shape) # [num_agent, vec_shape]
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
            cur_img_o.append(dog_map)
            
            cur_img_o = np.stack(cur_img_o, axis=0)
            cur_img_obs.append(cur_img_o)
        cur_img_obs = np.stack(cur_img_obs, axis=0)
        # import imageio
        # imageio.imwrite("../envs/mujoco/assets/map.png", cur_img_obs[0])
        # imageio.imwrite("../envs/mujoco/assets/map2.png", cur_img_obs[1])
                    
        # print("cur_img_obs.shape", cur_img_obs.shape) # [num_agent, 2, width, height]
        # # vector: global state 
        load_pos = self.sim.data.qpos.copy()[0:2].reshape([1,2])
        load_pos = (load_pos-self.goal.reshape([1,2])) / self.msize
        load_pos = self._cart2polar(load_pos)
        load_pos = np.repeat(load_pos, self.num_agent, axis=0)
        load_yaw = self.sim.data.qpos.copy()[2:3].reshape([1,1])
        load_sin = np.repeat(np.sin(load_yaw), self.num_agent, axis=0)
        load_cos = np.repeat(np.cos(load_yaw), self.num_agent, axis=0)
        load_yaw = [load_sin, load_cos]
        dog_state = self.sim.data.qpos.copy()[4:4+4*self.num_agent]
        dog_state = dog_state.reshape([self.num_agent, 4])
        dog_pos = (dog_state[:,0:2]-self.goal.reshape([1,2])) / self.msize
        dog_pos = self._cart2polar(dog_pos)
        dog_pos = dog_pos.reshape([1,-1])
        dog_pos = np.repeat(dog_pos, self.num_agent, axis=0)
        dog_yaw = dog_state[:,2:3].reshape([1,-1])
        dog_sin = np.repeat(np.sin(dog_yaw), self.num_agent, axis=0)
        dog_cos = np.repeat(np.cos(dog_yaw), self.num_agent, axis=0)
        dog_yaw = [dog_sin, dog_cos]
        # anchor_id = self.anchor_id.reshape([self.num_agent, 1])
        # anchor_vec = np.eye(4)[anchor_id].reshape([1, self.num_agent*4])
        # anchor_vec = np.repeat(anchor_vec, self.num_agent, axis=0)
        # cur_load_yaw = self.sim.data.qpos.copy()[2:3].reshape([1,1])
        # anchor_pos = self._get_toward(cur_load_yaw)[0] * 0.3 * (anchor_id==0)
        # anchor_pos += self._get_toward(cur_load_yaw)[1] * 0.3 * (anchor_id==1)
        # anchor_pos += self._get_toward(cur_load_yaw)[0] * (-0.3) * (anchor_id==2)
        # anchor_pos += self._get_toward(cur_load_yaw)[1] * (-0.3) * (anchor_id==3)
        # anchor_pos += self.sim.data.qpos.copy()[0:2]
        # rope_state = (np.linalg.norm(dog_state[:,0:2]-anchor_pos, axis=-1, keepdims=True)>1.)*1.
        cur_vec_sta = [load_pos, dog_pos] + load_yaw + dog_yaw
        cur_vec_sta = np.concatenate(cur_vec_sta, axis=-1)
        cur_vec_sta += (np.random.random(cur_vec_sta.shape)-.5) * self.measure_random_scale
        # print("cur_vec_sta.shape", cur_vec_sta.shape) # [num_agent, obs_shape]
        # iamge: global state
        cur_img_sta = []
        box_pos = self.sim.data.qpos.copy()[0:2]
        box_yaw = self.sim.data.qpos.copy()[2]
        obs_len = np.ones([2]) 
        obs_map = self._draw_map(
            box_pos, box_yaw, 
            self.init_obs_pos, self.init_obs_yaw, obs_len
        )
        wall_len = np.array([self.msize+2, 2.])
        obs_map += self._draw_map(
            box_pos, box_yaw, 
            self.wall_pos, self.wall_yaw, wall_len
        )
        cur_img_sta.append((obs_map!=0)*1.)
        
        all_dog_state = self.sim.data.qpos.copy()[4:4+4*self.num_agent]
        all_dog_state = all_dog_state.reshape([self.num_agent, 4])
        all_dog_pos = all_dog_state[:,0:2]
        all_dog_yaw = all_dog_state[:,2:3]
        dog_len = np.array([0.65, 0.3])
        dog_map = self._draw_map(
            box_pos, box_yaw, 
            all_dog_pos, all_dog_yaw, dog_len
        )
        cur_img_sta.append(dog_map)
        
        cur_img_sta = np.stack(cur_img_sta, axis=0)
        cur_img_o = np.concatenate(cur_img_obs, axis=0)
        cur_img_sta = np.concatenate([cur_img_sta, cur_img_o], axis=0)
        cur_img_sta = np.expand_dims(cur_img_sta, axis=0)
        cur_img_sta = np.repeat(cur_img_sta, self.num_agent, axis=0)
        
        # print("cur_img_sta.shape", cur_img_sta.shape) # [num_agent, 1+num_agent, width, height]
        # import imageio
        # img = np.concatenate([cur_img_sta[0,0], cur_img_sta[0,1]])
        # imageio.imwrite("../envs/mujoco/assets/map.png", img)

        return cur_vec_obs, cur_img_obs, cur_vec_sta, cur_img_sta
    
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

    # def _get_obs_map(self, obs_pos, obs_yaw):
    #     obs_toward, obs_vert = self._get_toward(obs_yaw)
    #     obs_vertice = self._get_rect(obs_pos, obs_toward, obs_vert, np.array([0.6,0.6]))
    #     hmlen = (self.lmlen*3-1) // 2
    #     obs_map = np.zeros([1, self.mlen+hmlen*2, self.mlen+hmlen*2])
    #     for rect in obs_vertice:
    #         norm_rect = (rect.copy() / self.msize + .5) * self.mlen
    #         clip_rect = np.clip(norm_rect.astype('long'), 0, self.mlen)
    #         min_i = np.min(clip_rect, axis=0) + hmlen
    #         max_i = np.max(clip_rect, axis=0) + hmlen
    #         obs_map[0,min_i[0]:max_i[0],min_i[1]:max_i[1]] += \
    #             self._draw_rect(norm_rect)
    #     obs_map = (obs_map!=0) * 1.
    #     obs_map[:,:hmlen+1] = 1.
    #     obs_map[:,:,:hmlen+1] = 1.
    #     obs_map[:,-hmlen-1:] = 1.
    #     obs_map[:,:,-hmlen-1:] = 1.
    #     return obs_map 

    def _do_simulation(self, action, n_frames):
        """ 
        (input) action: the desired velocity of the agent. shape=(num_agent, action_size)
        (input) n_frames: the agent will use this action for n_frames*dt seconds
        """
        for _ in range(n_frames):
            # assert np.array(ctrl).shape==self.action_space.shape, "Action dimension mismatch"
            self.sim.data.ctrl[:] = self._pd_contorl(target_vel=action, dt=self.dt/n_frames).flatten()
            self.sim.step()
            terminate, contact = self._get_done(self.dt/n_frames)
            if terminate:
                break
        return terminate, contact

    def _pd_contorl(self, target_vel, dt):
        """
        given the desired velocity, calculate the torque of the actuator.
        (input) target_vel: 
        (input) dt: 
        (output) torque
        """
        state = self.sim.data.qvel.copy()[4:4+self.num_agent*4]
        cur_vel = state.reshape([self.num_agent, 4])[:,:3]
        error = target_vel - cur_vel
        self.intergral += error * self.ki * dt
        self.intergral = np.clip(self.intergral, self.torque_low, self.torque_high)
        d_output = cur_vel - self.prev_output_vel
        torque = self.kp * error - self.kd * (d_output/dt) + self.intergral
        torque = np.clip(torque, self.torque_low, self.torque_high)
        torque_norm = np.linalg.norm(torque[:,:2], axis=-1)
        for i in range(len(torque_norm)):
            if torque_norm[i] > self.max_torque:
                torque[i,:2] = torque[i,:2] * self.max_torque / torque_norm[i] 
        self.prev_output_vel = cur_vel.copy()
        return torque

    def _post_update(self, cmd, cur_obs):
        """
        restore the history information
        """
        cur_vec_obs, cur_img_obs, cur_vec_sta, cur_img_sta = cur_obs
        # update hist info
        self.hist_vec_obs[:,:-1] = self.hist_vec_obs[:,1:]
        cur_vec_obs = np.concatenate([cur_vec_obs, cmd], -1)
        self.hist_vec_obs[:,-1] = cur_vec_obs.copy()
        self.hist_img_obs[:,:-1] = self.hist_img_obs[:,1:]
        self.hist_img_obs[:,-1] = cur_img_obs.copy()
        self.hist_vec_sta[:,:-1] = self.hist_vec_sta[:,1:]
        cmd = cmd.reshape([1,-1])
        cmd = np.repeat(cmd, self.num_agent, 0)
        cur_vec_sta = np.concatenate([cur_vec_sta, cmd], -1)
        self.hist_vec_sta[:,-1] = cur_vec_sta.copy()
        self.hist_img_sta[:,:-1] = self.hist_img_sta[:,1:]
        self.hist_img_sta[:,-1] = cur_img_sta.copy()
        # update last distance 
        load_pos = self.sim.data.qpos.copy()[0:2]
        self.prev_dist = np.linalg.norm(load_pos-self.goal)
        self.last_load_pos = load_pos.copy()

    def _local_to_global(self, input_action):

        state = self.sim.data.qpos.flat.copy()[4:4+self.num_agent*4]
        state = state.reshape([self.num_agent,4])
        tow, ver = self._get_toward(state[:,2:3])
        input_action = input_action.reshape([self.num_agent,3])
        local_action = input_action[:,:2].copy()
        global_action = tow * local_action[:,:1] + ver * local_action[:,1:]
        output_action = np.concatenate([global_action, input_action[:,2:].copy()], axis=-1)
        return output_action
        
    # def _get_cost_map(self, obstacle_map):
    #     hmlen = (self.lmlen*3-1) // 2
    #     cost_map = np.zeros_like(obstacle_map[:,hmlen:-hmlen,hmlen:-hmlen])
    #     obs_map = obstacle_map[:,hmlen:-hmlen,hmlen:-hmlen].astype('bool')
    #     times = 35
    #     for _ in range(times):
    #         tmp_map = obs_map.copy().astype('bool')
    #         obs_map[:,:-1,:] |= tmp_map[:,1:,:]
    #         obs_map[:,1:,:] |= tmp_map[:,:-1,:]
    #         obs_map[:,:,:-1] |= tmp_map[:,:,1:]
    #         obs_map[:,:,1:] |= tmp_map[:,:,:-1]
    #         cost_map += obs_map / times
    #     return cost_map
    