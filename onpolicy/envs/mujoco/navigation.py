import numpy as np
from os import path
from gym import utils
from onpolicy.envs.mujoco.base_env import BaseEnv
from onpolicy.envs.mujoco.xml_gen import get_xml
from gym.spaces import Box, Tuple
import time

class NavigationEnv(BaseEnv):
    def __init__(self, **kwargs):
        # hyper-para
        self.lmlen = 57 # local map length (pixels)
        self.warm_step = 4 # warm-up: let everything stable  (s)
        self.msize = 10. # map size (m)
        self.mlen = 400 # global map length (pixels)
        self.random_scale = 0.01 # add noise to the observed coordinate (m)
        self.frame_skip = 25
        self.num_agent = 1
        self.num_obs = 10
        self.hist_len = 4
        self.max_cont_time = 1
        self.encoding_num = 4
        self.init_kp = np.array([[2000, 2000, 80]])
        self.init_kd = np.array([[0.02, 0.02, 0.01]])
        self.init_ki = np.array([[0.0, 0.0, 10.]])
        # simulator
        model = get_xml(dog_num=self.num_agent, obs_num=self.num_obs)
        super().__init__(model, **kwargs)
        # observation space 
        self.observation_space = Tuple((
            Box(low=-np.inf, high=np.inf, shape=(178,), dtype=np.float64), # 42 234
            Box(low=-np.inf, high=np.inf, shape=(4,self.lmlen,self.lmlen), dtype=np.float64),
        ))
        # action space
        aspace_low = np.array([-0.25, -0.05, -0.5])
        aspace_high = np.array([0.5, 0.05, 0.5])
        self.action_space = Box(
            low=aspace_low, high=aspace_high, shape=(3,), dtype=np.float64
        )
        self.action_mean = (aspace_high+aspace_low)/2
        self.action_range = aspace_high-aspace_low

        # hyper-para
        bounds = self.model.actuator_ctrlrange.copy()
        self.torque_low, self.torque_high = bounds.astype(np.float32).T
        # variables
        self.reuse_init_state = False
        self.init_state = None

    def seed(self, seed):
        super().seed(seed)
        np.random.seed(seed)

    def reset(self):
        # init random tasks
        self.kp  = self.init_kp * (1. + (np.random.random(3)-.5) * 0.)
        self.ki  = self.init_ki * (1. + (np.random.random(3)-.5) * 0.)
        self.kd  = self.init_kd * (1. + (np.random.random(3)-.5) * 0.)
        # init variables
        self.t = 0.
        self.max_time = 1e6
        self.cont_time = self.max_cont_time-1
        self.last_cmd = np.zeros([self.num_agent, self.action_space.shape[0]])
        self.intergral = np.zeros([self.num_agent, self.action_space.shape[0]])
        self.hist_vec_obs = np.zeros([self.hist_len-1, self.num_agent, 45]) # 11 57
        self.hist_img_obs = np.zeros([self.hist_len-1, self.num_agent, *self.observation_space[1].shape[-2:]])
        self.prev_output = np.zeros([self.num_agent, self.action_space.shape[0]])
        # regenerate env
        if self.reuse_init_state and np.random.rand() < 0.:
            qpos = self.init_state.copy()
            self.reuse_init_state = False
        else:
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
            init_dog_load_yaw = np.random.random([self.num_agent, 1]) * np.pi * 2
            init_dog_yaw = init_load_yaw + (np.random.random([self.num_agent, 1])-.5) * np.pi
            init_dog_pos = init_load_pos.reshape([1,2])
            init_dog_pos += self._get_toward(init_dog_load_yaw)[0] * init_dog_load_len
            init_dog_pos += self._get_toward(init_load_yaw)[0] * 0.3
            init_dog_z = np.ones([self.num_agent, 1]) * 0.3
            init_dog = np.concatenate([init_dog_pos, init_dog_yaw, init_dog_z], axis=-1).flatten()
            self.goal = (np.random.random(2)-.5) * (self.msize-2.)
            qpos = np.concatenate([init_load, init_dog, init_obs, init_wall, self.goal.flatten()])
            self.init_state = qpos.copy()
        self.set_state(np.array(qpos), np.zeros_like(qpos))
        for _ in range(self.warm_step):
            terminated, _ = self._do_simulation(self.last_cmd.copy(), self.frame_skip)
            # regenerate if done
            if terminated:
                self.reuse_init_state = False
                return self.reset()
        # init variables
        self.cont_time = 0
        load_pos = self.sim.data.qpos.copy()[:2]
        dist = np.linalg.norm(load_pos-self.goal, axis=-1)
        self.max_time = np.clip(dist*4.+10., 5., 1e6)
        self.t = 0.
        self.obs_map = self._get_obs_map(self.init_obs_pos, self.init_obs_yaw)
        # RL_info
        cur_obs, observation  = self._get_obs()
        info = dict()
        # post process
        self._post_update(self.last_cmd, cur_obs)
        return observation, info

    def step(self, cmd):
        # pre process
        command = np.clip(cmd, self.action_space.low, self.action_space.high)
        action = self._local_to_global(command)
        # action = command.copy()
        terminated, contact = self._do_simulation(action, self.frame_skip)
        self.t += self.dt
        # update RL info
        cur_obs, observation = self._get_obs()
        reward, info = self._get_reward(contact)
        # post process
        self._post_update(command, cur_obs)
        return observation, reward, terminated, False, info

    def _get_done(self): 
        terminate = False
        contact = False
        # contact done
        for i in range(self.sim.data.ncon):
            con = self.sim.data.contact[i]
            obj1 = self.sim.model.geom_id2name(con.geom1)
            obj2 = self.sim.model.geom_id2name(con.geom2)
            if obj1=="floor" or obj2=="floor":
                continue
            self.cont_time += 1
            contact = True
            if self.cont_time < self.max_cont_time:
                continue
            terminate = True
            self.reuse_init_state = True
            return terminate, contact
        # out of time
        if self.t > self.max_time:
            terminate = True
            contact = False
            # state = self.sim.data.qpos.copy().flatten()
            # if np.linalg.norm(state[:2]-self.goal) > 0.5:
            #     self.reuse_init_state = True
            return terminate, contact
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
        rewards.append((dist<0.5) * max_speed * duration)
        # done penalty
        rewards.append(-contact*1.)
        # print(rewards)
        rew_dict = dict()
        rew_dict["goal_distance"] = rewards[0]
        rew_dict["goal_reach"] = rewards[1]
        rew_dict["con_penalty"] = rewards[2]
        return np.dot(np.array(rewards), weights), rew_dict

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

    def _get_obs_map(self, obs_pos, obs_yaw):
        obs_toward, obs_vert = self._get_toward(obs_yaw)
        obs_vertice = self._get_rect(obs_pos, obs_toward, obs_vert, np.array([0.6,0.6]))
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
        ever_contact = False
        for _ in range(n_frames):
            ctrl = self._pd_contorl(target_vel=action, dt=self.dt/n_frames).flatten()
            assert np.array(ctrl).shape==self.action_space.shape, "Action dimension mismatch"
            self.sim.data.ctrl[:] = ctrl
            self.sim.step()
            terminate, contact = self._get_done()
            ever_contact |= contact
            if terminate:
                break
        return terminate, ever_contact

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
        d_output = cur_vel - self.prev_output
        torque = self.kp * error - self.kd * (d_output/dt) + self.intergral
        torque = np.clip(torque, self.torque_low, self.torque_high)
        self.prev_output = cur_vel.copy()
        return torque

    def _post_update(self, cmd, cur_obs):
        """
        restore the history information
        """
        cur_vec_obs = self._get_cur_vec_obs(cmd.reshape([self.num_agent, -1]))
        # cur_vec_obs += self.random_scale * (np.random.random(cur_vec_obs.shape)-.5)
        self.hist_vec_obs[:-1] = self.hist_vec_obs[1:]
        self.hist_vec_obs[-1] = cur_vec_obs
        self.hist_img_obs[:-1] = self.hist_img_obs[1:]
        self.hist_img_obs[-1] = cur_obs[1].copy()
        load_pos = self.sim.data.qpos.copy()[0:2]
        self.prev_dist = np.linalg.norm(load_pos-self.goal)

    def _get_cur_vec_obs(self, cmd=None):

        load_pos = self.sim.data.qpos.copy()[0:2].reshape([1,2])
        load_pos = (load_pos-self.goal.reshape([1,2])) / self.msize
        load_pos_enc = [np.sin(2**i*np.pi*load_pos) for i in range(self.encoding_num)]
        load_pos_enc += [np.cos(2**i*np.pi*load_pos) for i in range(self.encoding_num)]
        load_yaw = self.sim.data.qpos.copy()[2:3].reshape([1,1])
        load_yaw_enc = [np.sin(2**i*load_yaw) for i in range(1)]
        load_yaw_enc += [np.cos(2**i*load_yaw) for i in range(1)]
        dog_state = self.sim.data.qpos.copy()[4:4+4*self.num_agent]
        dog_state = dog_state.reshape([self.num_agent, 4])
        dog_pos = (dog_state[:,0:2]-self.goal.reshape([1,2])) / self.msize
        dog_load_pos = (dog_state[:,0:2]-load_pos) 
        dog_pos_enc = [np.sin(2**i*np.pi*dog_pos) for i in range(self.encoding_num)]
        dog_pos_enc += [np.cos(2**i*np.pi*dog_pos) for i in range(self.encoding_num)]
        dog_yaw = dog_state[:,2:3]
        dog_yaw_enc = [np.sin(2**i*dog_yaw) for i in range(1)]
        dog_yaw_enc += [np.cos(2**i*dog_yaw) for i in range(1)]
        vec_obs = [load_pos, dog_pos, dog_load_pos]
        vec_obs += (load_yaw_enc + load_pos_enc + dog_yaw_enc + dog_pos_enc)
        if cmd is not None:
            vec_obs.append(np.array(cmd))
        return np.concatenate(vec_obs, axis=-1)


    def _get_obs(self):
        
        # vec_obs
        cur_vec_obs = self._get_cur_vec_obs()
        # cur_vec_obs += self.random_scale * (np.random.random(cur_vec_obs.shape)-.5)
        hist_obs = self.hist_vec_obs.copy()
        hist_obs = hist_obs.transpose(1,0,2)
        hist_obs = hist_obs.reshape([self.num_agent, -1])
        time_obs = np.array([[self.t] for _ in range(self.num_agent)]) / 20.
        vec_observation = np.concatenate([hist_obs, cur_vec_obs, time_obs], axis=-1)
        # img_obs
        cur_img_obs = []
        for i in range(self.num_agent):
            idx = 4 * (i+1)
            dog_pos = self.sim.data.qpos.copy()[idx:idx+2]
            coor = ((dog_pos/self.msize+.5)*self.mlen).astype('long')
            x1, x2 = coor[0], coor[0]+self.lmlen*3
            y1, y2 = coor[1], coor[1]+self.lmlen*3
            local_map = self.obs_map[0, x1:x2, y1:y2]
            zeros_map = np.zeros([self.lmlen*3,self.lmlen*3])
            if (np.array(local_map.shape) != np.array(zeros_map.shape)).any():
                local_map = np.zeros([self.lmlen,self.lmlen])
            else:
                local_map = \
                    local_map[0::3,0::3] + local_map[1::3,0::3] + local_map[2::3,0::3] + \
                    local_map[0::3,1::3] + local_map[1::3,1::3] + local_map[2::3,1::3] + \
                    local_map[0::3,2::3] + local_map[1::3,2::3] + local_map[2::3,2::3] 
            cur_img_obs.append((local_map>0.)*1.)
        cur_img = np.expand_dims(cur_img_obs, 1)
        hist_img = self.hist_img_obs.copy()
        hist_img = hist_img.transpose(1,0,2,3)
        img_observation = np.concatenate([cur_img, hist_img], axis=1)

        # all_obs
        cur_obs = (cur_vec_obs, cur_img_obs)
        observation = (vec_observation, img_observation)
        # print(vec_observation.shape)
        # import imageio
        # imageio.imwrite("../envs/mujoco/assets/map.png",img_observation[0,0])
        return cur_obs, observation
    
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
    