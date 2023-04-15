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
        self.mlen = 400 # global map length (pixels)
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
        self.load_mass = 1. * (1 + (np.random.rand()-.5) * self.domain_random_scale)
        self.cable_len = 1. * (1 + (np.random.random(self.num_agent)-.5) * self.domain_random_scale)
        self.anchor_id = np.random.randint(0, 4, self.num_agent)
        model = get_xml(
            dog_num = self.num_agent, 
            obs_num = self.num_obs, 
            anchor_id = self.anchor_id,
            load_mass = self.load_mass,
            cable_len = self.cable_len,
        )
        super().__init__(model, **kwargs)
        # observation space 
        obs_size = 12 + 11 * (self.hist_len-1) 
        self.observation_space = Tuple((
            Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64), 
            Box(low=-np.inf, high=np.inf, shape=(8,self.lmlen,self.lmlen), dtype=np.float64),
        ))
        sta_size = 5 + 9*self.num_agent + (4+7*self.num_agent) * (self.hist_len-1)
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

    def seed(self, seed):
        super().seed(seed)
        np.random.seed(seed)

    def reset(self):
        # init random tasks
        np.random.seed(100)
        self.kp  = self.init_kp * (1. + (np.random.random(3)-.5) * self.domain_random_scale)
        self.ki  = self.init_ki * (1. + (np.random.random(3)-.5) * self.domain_random_scale)
        self.kd  = self.init_kd * (1. + (np.random.random(3)-.5) * self.domain_random_scale)
        # init variables
        self.t = 0.
        self.max_time = 1e6
        self.last_cmd = np.zeros([self.num_agent, self.action_space.shape[0]])
        self.intergral = np.zeros([self.num_agent, self.action_space.shape[0]])
        self.hist_vec_obs = np.zeros([self.num_agent, self.hist_len-1, 11]) 
        self.hist_img_obs = np.zeros([self.num_agent, self.hist_len-1, 2, *self.observation_space[1].shape[-2:]])
        self.hist_vec_sta = np.zeros([self.num_agent, self.hist_len-1, 4+7*self.num_agent]) 
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
        load_pos = self.sim.data.qpos.copy()[:2]
        dist = np.linalg.norm(load_pos-self.goal, axis=-1)
        self.max_time = dist * 10. + 30.
        self.obs_map = self._get_map(self.init_obs_pos, self.init_obs_yaw)
        # RL_info
        self.cul_rew = 0.
        cur_obs, observation = self._get_obs() 
        info = dict()
        # post process
        self._post_update(self.last_cmd, cur_obs)
        return observation, info
    
    def _get_obs(self):
        
        dog_state = self.sim.data.qpos.copy()[4:4+4*self.num_agent].reshape([-1,4])
        dog_pos = dog_state[:,:2]
        dog_yaw = dog_state[:,2:3]
        self.dog_map = self._get_map(dog_pos, dog_yaw, wall=False)
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
        self.cul_rew += reward
        # post process
        self._post_update(command, cur_obs)
        return observation, reward, done, False, info

    def _get_done(self): 

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
        
        return terminate, contact

    def _get_reward(self, contact):

        rewards = []
        # pre-process
        weights = np.array([0., 2., 1., 0.5, 0.])
        weights = weights / weights.sum()
        state = self.sim.data.qpos.copy().flatten()
        # goal_distance rewards
        dist = np.linalg.norm(state[:2]-self.goal)
        rewards.append((self.prev_dist-dist)*(dist>0.5))
        # goal reach rewards
        max_speed = np.linalg.norm(self.action_space.high[:2])
        duration = self.frame_skip / 100
        rewards.append((dist<0.5) * max_speed * duration)
        # done penalty
        rewards.append(-contact*1.)
        # dog move rew
        dist = np.linalg.norm(state[4:6]-self.goal)
        rewards.append((self.prev_dog_dist-dist)*(dist>0.5))
        # dog yaw
        dog_box_vec = self.goal - state[4:6]
        dog_box_yaw = np.arctan2(dog_box_vec[1], dog_box_vec[0])
        rewards.append(np.cos(dog_box_yaw-state[6])+1)

        rew_dict = dict()
        rew_dict["goal_distance"] = rewards[0]
        rew_dict["goal_reach"] = rewards[1]
        rew_dict["con_penalty"] = rewards[2]
        rew_dict["dog_move"] = rewards[3]
        rew_dict["dog_yaw"] = rewards[4]
        rews = np.dot(np.array(rewards), weights)
        return rews, rew_dict

    def _get_cur_obs(self):

        # vector: partial observation
        load_pos = self.sim.data.qpos.copy()[0:2].reshape([1,2])
        load_pos = (load_pos-self.goal.reshape([1,2])) / self.msize
        load_pos = np.repeat(load_pos, self.num_agent, axis=0)
        load_yaw = self.sim.data.qpos.copy()[2:3].reshape([1,1])
        load_yaw = np.repeat(load_yaw, self.num_agent, axis=0)
        load_yaw = [np.sin(load_yaw), np.cos(load_yaw)]
        dog_state = self.sim.data.qpos.copy()[4:4+4*self.num_agent]
        dog_state = dog_state.reshape([self.num_agent, 4])
        dog_pos = (dog_state[:,0:2]-self.goal.reshape([1,2])) / self.msize
        dog_yaw = dog_state[:,2:3]
        dog_yaw = [np.sin(dog_yaw), np.cos(dog_yaw)]
        # anchor_id = self.anchor_id.reshape([self.num_agent, 1])
        # anchor_vec = np.eye(4)[anchor_id][:,0]
        cur_vec_obs = [load_pos, dog_pos] + load_yaw + dog_yaw
        cur_vec_obs = np.concatenate(cur_vec_obs, axis=-1) 
        cur_vec_obs += (np.random.random(cur_vec_obs.shape)-.5) * self.measure_random_scale
        # print("cur_vec_obs.shape", cur_vec_obs.shape) # [num_agent, vec_shape]
        # image: partial observation
        cur_img_obs = []
        for i in range(self.num_agent):
            dog_pos = self.sim.data.qpos.copy()[4+4*i:6+4*i]
            coor = ((dog_pos/self.msize+.5)*self.mlen).astype('long')
            x1, x2 = coor[0], coor[0]+self.lmlen*3
            y1, y2 = coor[1], coor[1]+self.lmlen*3
            local_obs_map = self.obs_map[0, x1:x2, y1:y2]
            local_dog_map = self.dog_map[0, x1:x2, y1:y2]
            zeros_map = np.zeros([self.lmlen*3,self.lmlen*3])
            if (np.array(local_obs_map.shape) != np.array(zeros_map.shape)).any(): # TODO: simplified
                local_obs_map = np.zeros([self.lmlen,self.lmlen])
                local_dog_map = np.zeros([self.lmlen,self.lmlen])
            else: # down-sampling
                local_obs_map = \
                    local_obs_map[0::3,0::3] + local_obs_map[1::3,0::3] + local_obs_map[2::3,0::3] + \
                    local_obs_map[0::3,1::3] + local_obs_map[1::3,1::3] + local_obs_map[2::3,1::3] + \
                    local_obs_map[0::3,2::3] + local_obs_map[1::3,2::3] + local_obs_map[2::3,2::3] 
                local_dog_map = \
                    local_dog_map[0::3,0::3] + local_dog_map[1::3,0::3] + local_dog_map[2::3,0::3] + \
                    local_dog_map[0::3,1::3] + local_dog_map[1::3,1::3] + local_dog_map[2::3,1::3] + \
                    local_dog_map[0::3,2::3] + local_dog_map[1::3,2::3] + local_dog_map[2::3,2::3] 
            cur_img_obs.append([(local_obs_map>0.)*1.,(local_dog_map>0.)*1.])
        cur_img_obs = np.array(cur_img_obs)
        # cur_img_obs = np.expand_dims(cur_img_obs, axis=1)
        # print("cur_img_obs.shape", cur_img_obs.shape) # [num_agent, 2, width, height]
        # vector: global state 
        load_pos = self.sim.data.qpos.copy()[0:2].reshape([1,2])
        load_pos = (load_pos-self.goal.reshape([1,2])) / self.msize
        load_pos = np.repeat(load_pos, self.num_agent, axis=0)
        load_yaw = self.sim.data.qpos.copy()[2:3].reshape([1,1])
        load_sin = np.repeat(np.sin(load_yaw), self.num_agent, axis=0)
        load_cos = np.repeat(np.cos(load_yaw), self.num_agent, axis=0)
        load_yaw = [load_sin, load_cos]
        dog_state = self.sim.data.qpos.copy()[4:4+4*self.num_agent]
        dog_state = dog_state.reshape([self.num_agent, 4])
        dog_pos = (dog_state[:,0:2]-self.goal.reshape([1,2])) / self.msize
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
        load_pos = self.sim.data.qpos.copy()[:2]
        coor = ((load_pos/self.msize+.5)*self.mlen).astype('long')
        x1, x2 = coor[0], coor[0]+self.lmlen*3
        y1, y2 = coor[1], coor[1]+self.lmlen*3
        local_map = self.obs_map[0, x1:x2, y1:y2]
        local_dog_map = self.dog_map[0, x1:x2, y1:y2]
        zeros_map = np.zeros([self.lmlen*3,self.lmlen*3])
        if (np.array(local_map.shape) != np.array(zeros_map.shape)).any(): # TODO: simplified
            local_map = np.zeros([self.lmlen,self.lmlen])
            local_dog_map = np.zeros([self.lmlen,self.lmlen])
        else: # down-sampling
            local_map = \
                local_map[0::3,0::3] + local_map[1::3,0::3] + local_map[2::3,0::3] + \
                local_map[0::3,1::3] + local_map[1::3,1::3] + local_map[2::3,1::3] + \
                local_map[0::3,2::3] + local_map[1::3,2::3] + local_map[2::3,2::3] 
            local_dog_map = \
                local_dog_map[0::3,0::3] + local_dog_map[1::3,0::3] + local_dog_map[2::3,0::3] + \
                local_dog_map[0::3,1::3] + local_dog_map[1::3,1::3] + local_dog_map[2::3,1::3] + \
                local_dog_map[0::3,2::3] + local_dog_map[1::3,2::3] + local_dog_map[2::3,2::3] 
        cur_img_sta = np.stack([(local_map>0.)*1., (local_dog_map>0.)*1.], axis=0)
        # cur_img_sta = np.expand_dims(cur_img_sta, axis=0) # [2, width, height]
        tmp_img_obs = cur_img_obs.reshape([-1, *cur_img_obs.shape[-2:]])
        cur_img_sta = np.concatenate([cur_img_sta, tmp_img_obs], 0)
        cur_img_sta = np.expand_dims(cur_img_sta, axis=0)
        cur_img_sta = np.repeat(cur_img_sta, self.num_agent, axis=0)
        # print("cur_img_sta.shape", cur_img_sta.shape) # [num_agent, 1+num_agent, width, height]
        import imageio
        imageio.imwrite("../envs/mujoco/assets/map.png", cur_img_obs[0,1])
        return cur_vec_obs, cur_img_obs, cur_vec_sta, cur_img_sta

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
        x_axis = np.arange(length[0]) + .5 + min_idx[0]
        y_axis = np.arange(length[1]) + .5 + min_idx[1]
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

    def _get_map(self, obs_pos, obs_yaw, wall=True):
        obs_toward, obs_vert = self._get_toward(obs_yaw)
        shape = np.array([0.6,0.6]) if wall else np.array([0.65,0.3])
        obs_vertice = self._get_rect(obs_pos, obs_toward, obs_vert, shape)
        hmlen = (self.lmlen*3-1) // 2
        obs_map = np.zeros([1, self.mlen+hmlen*2, self.mlen+hmlen*2])
        for rect in obs_vertice:
            norm_rect = (rect.copy() / self.msize + .5) * self.mlen
            clip_rect = np.clip(norm_rect.astype('long'), 0, self.mlen)
            min_i = np.min(clip_rect, axis=0) + hmlen
            max_i = np.max(clip_rect, axis=0) + hmlen
            obs_map[0,min_i[0]:max_i[0],min_i[1]:max_i[1]] += \
                self._draw_rect(norm_rect, min_i-hmlen, max_i-hmlen)
        obs_map = (obs_map!=0) * 1.
        if wall:
            obs_map[:,:hmlen+1] = 1.
            obs_map[:,:,:hmlen+1] = 1.
            obs_map[:,-hmlen-1:] = 1.
            obs_map[:,:,-hmlen-1:] = 1.
        return obs_map 

    def _do_simulation(self, action, n_frames):
        """ 
        (input) action: the desired velocity of the agent. shape=(num_agent, action_size)
        (input) n_frames: the agent will use this action for n_frames*dt seconds
        """
        for _ in range(n_frames):
            # assert np.array(ctrl).shape==self.action_space.shape, "Action dimension mismatch"
            self.sim.data.ctrl[:] = self._pd_contorl(target_vel=action, dt=self.dt/n_frames).flatten()
            self.sim.step()
            terminate, contact = self._get_done()
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
        dog_pos = self.sim.data.qpos.copy()[4:6]
        self.prev_dog_dist = np.linalg.norm(dog_pos-self.goal)

    def _local_to_global(self, input_action):

        state = self.sim.data.qpos.flat.copy()[4:4+self.num_agent*4]
        state = state.reshape([self.num_agent,4])
        tow, ver = self._get_toward(state[:,2:3])
        input_action = input_action.reshape([self.num_agent,3])
        local_action = input_action[:,:2].copy()
        global_action = tow * local_action[:,:1] + ver * local_action[:,1:]
        output_action = np.concatenate([global_action, input_action[:,2:].copy()], axis=-1)
        return output_action
        

    