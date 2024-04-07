import numpy as np
from os import path
from gym import utils
from onpolicy.envs.mujoco.base_env import BaseEnv
from onpolicy.envs.mujoco.xml_gen import get_xml
from onpolicy.envs.mujoco.astar import find_path
from gym.spaces import Box, Tuple
import time

class NavigationEnv(BaseEnv):
    def __init__(self, num_agents, env_rank, **kwargs):
        
        self._kwargs = kwargs
        
        # hyper-para
        self.env_rank = env_rank
        self.previledge_critic = True
        self.regenerate_ratio = 0.75
        self.arrive_dist = 1.
        self.mlen = 500 # global map length (pixels)
        self.msize = 10. # map size (m)
        self.lmlen = 57 # local map length (pixels)
        self.warm_step = 4 # warm-up: let everything stable (s)
        self.frame_skip = 10 # 100/frame_skip = decision_freq (Hz)
        self.num_obs = 10
        self.hist_len = 4
        self.num_agent = num_agents
        self.domain_random_scale = 1e-1
        self.measure_random_scale = 1e-2
        self.init_kp = np.array([[2000, 2000, 800]])
        self.init_kd = np.array([[0.02, 0.02, 1e-6]])
        # self.init_ki = np.array([[0.0, 0.0, 10.]])
        self.max_axis_torque = 100.
        self.astar_node = 20 # for rendering astar path 
        self.box_half_len = 0.25
        self.arrive_thresh = 0.2
        self.stuck_time = 5.
        self.gamma = 0.99
        # simulator
        load_mass = np.array([0., 3., 5., 5., 5.])[self.num_agent]
        self.load_mass = load_mass * np.clip(np.sqrt(np.random.rand()), 0.2, 1.)
        self.cable_len = 1. * (1 + (np.random.random(self.num_agent)-.5) * self.domain_random_scale)
        self.anchor_id = (np.array([3, 0, 1]) + self.env_rank) #% 4
        # if self.env_rank % 10 < 4:
        #     self.anchor_id = np.zeros(2) + self.env_rank % 10
        # elif self.env_rank % 10 < 8:
        #     self.anchor_id = (np.arange(2) + self.env_rank % 10) % 4
        # else:
        #     self.anchor_id = (np.array([0,2]) + self.env_rank % 10) % 4
        self.fric_coef = 1. * (1 + (np.random.random(self.num_agent)-.5) * self.domain_random_scale)
        model = get_xml(
            dog_num = self.num_agent, 
            obs_num = self.num_obs, 
            anchor_id = self.anchor_id,
            load_mass = self.load_mass,
            cable_len = self.cable_len,
            fric_coef = self.fric_coef,
            astar_node = self.astar_node,
            box_half_len = self.box_half_len,
        )
        super().__init__(model, **kwargs)
        
        # observation space 
        self.hist_vec_obs_size = 13
        obs_size = self.hist_vec_obs_size # * self.hist_len
        self.observation_space = Tuple((
            Box(low=-np.inf, high=np.inf, shape=(obs_size + 3,), dtype=np.float64), 
            Box(low=-np.inf, high=np.inf, shape=(3, self.lmlen, self.lmlen), dtype=np.float64),
        ))
        sta_size = (obs_size + 10) * self.num_agent + 11
        sta_size = sta_size if self.previledge_critic else obs_size*self.num_agent
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
        self.goal = np.zeros([2]) + 1e6
        self.env_idx = 0
        
        self.arrive_time = 0.
        self.total_step = 0
        self.hist_load = []
        self.env_data = dict()

    def seed(self, seed):
        super().seed(seed)
        np.random.seed(int(time.time()))

    def reset(self):
        
        # init random tasks
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
        self.order = np.arange(self.num_agent)
        # self.env_idx = np.random.randint(69)
        # if self.env_rank % 10 < 4:
        #     self.env_idx = np.random.randint(87)
        # elif self.env_rank % 10 < 8:
        #     self.env_idx = np.random.randint(76) + 87
        # else:
        #     self.env_idx = np.random.randint(36) + 163
        
        if False: # self.env_idx in self.env_data:
            import json 
            if self.env_idx not in self.env_data:
                data_path = "./results/3agent/{:04d}.json".format(self.env_idx)
                with open(data_path) as file:
                    data = json.load(file)
                self.env_data[self.env_idx] = data
            qpos = np.array(self.env_data[self.env_idx]['qpos'])
            qpos[2] -= np.pi/2 * self.anchor_id[0]
            self.goal = np.array(self.env_data[self.env_idx]['goal'])
            self.path = self.env_data[self.env_idx]['path4render']
            self.astar_path = np.array(self.env_data[self.env_idx]['astar_path'])
            self.init_obs_pos = np.array(self.env_data[self.env_idx]['init_obs_pos'])
            self.init_obs_yaw = np.array(self.env_data[self.env_idx]['init_obs_yaw'])
            self.set_state(np.array(qpos), np.zeros_like(qpos))
            self._do_simulation(self.last_cmd.copy(), self.frame_skip)
        else:
            init_load_pos = (np.random.random(2)-.5) * self.msize 
            init_load_yaw = np.random.random(1) * 2 * np.pi 
            # init_load_pos = np.array([0.,0.])
            # init_load_yaw = np.array([0.])
            init_load_z = np.ones(1) * 0.55
            init_load = np.concatenate([init_load_pos, init_load_yaw, init_load_z], axis=-1).flatten()
            min_dist = 0.
            while min_dist < 0.6:
                init_dog_load_len = np.random.random([self.num_agent, 1]) * 0.25 + 0.75
                init_dog_load_yaw = (np.random.random([self.num_agent, 1])-.5) * np.pi
                init_dog_yaw = np.random.random([self.num_agent, 1]) * np.pi * 2
                # init_dog_load_len = np.array([[.75],[.75]])
                # init_dog_load_yaw = np.array([[0.],[0.]])
                # init_dog_yaw = np.array([[0.],[0.]])
                anchor_id = self.anchor_id.reshape([self.num_agent, 1])
                anchor_pos = self._get_toward(init_load_yaw)[0] * self.box_half_len * (anchor_id==0)
                anchor_pos += self._get_toward(init_load_yaw)[1] * self.box_half_len * (anchor_id==1)
                anchor_pos += self._get_toward(init_load_yaw)[0] * (-self.box_half_len) * (anchor_id==2)
                anchor_pos += self._get_toward(init_load_yaw)[1] * (-self.box_half_len) * (anchor_id==3)
                anchor_pos += init_load_pos.reshape([1,2])
                anc_dog_yaw = init_load_yaw + anchor_id * np.pi/2 + init_dog_load_yaw
                init_dog_pos = self._get_toward(anc_dog_yaw)[0] * init_dog_load_len
                init_dog_pos += anchor_pos
                init_dog_z = np.ones([self.num_agent, 1]) * self.box_half_len
                dog2dog = init_dog_pos.reshape([-1,1,2])-init_dog_pos.reshape([1,-1,2])
                min_dist = np.linalg.norm(dog2dog, axis=-1) + np.eye(self.num_agent)
                min_dist = min_dist.min()        
                
            init_dog = np.concatenate([init_dog_pos, init_dog_yaw, init_dog_z], axis=-1).flatten()
            idx = 4+4*self.num_agent+4*self.num_obs
            init_wall = self.sim.data.qpos.copy()[idx:idx+12]
            min_dist = 0.
            while min_dist < .8:
                init_obs_z = np.ones([self.num_obs, 1]) * 0.55
                self.init_obs_pos = (np.random.random([self.num_obs, 2])-.5) * self.msize 
                self.init_obs_yaw = np.random.random([self.num_obs, 1]) * np.pi
                # self.init_obs_pos = np.array([[3.,2.]])
                # self.init_obs_yaw = np.array([[np.pi/2]])
                load_dist = np.linalg.norm(self.init_obs_pos-init_load_pos, axis=-1).min()
                d_pos = init_dog_pos.reshape([1,-1,2])
                o_pos = self.init_obs_pos.reshape([-1, 1, 2])
                obs2dog = (o_pos-d_pos).reshape([-1,2])
                dog_dist = np.linalg.norm(obs2dog, axis=-1).min()
                min_dist = min(load_dist, dog_dist)
            init_obs = np.concatenate([self.init_obs_pos, self.init_obs_yaw, init_obs_z], axis=-1).flatten()
            min_dist, load_goal_dist = 0., 1e6
            while min_dist < 0.8 or load_goal_dist > 7.:
                self.goal = (np.random.random(2)-.5) * (self.msize-2.)
                # self.goal = np.array([4.,4.])
                min_dist = np.linalg.norm(self.goal.reshape([1,2])-self.init_obs_pos, axis=-1).min()
                load_goal_dist = np.linalg.norm(self.goal-init_load_pos)
                    
            qpos = np.concatenate([init_load, init_dog, init_obs, init_wall, np.zeros(self.astar_node*2+2)])
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
            
            self.astar_path = find_path(load_pos, goal_pos, obs_map)
            
            if self.astar_path is None:
                return self.reset()
            
            straight = True
            for i in range(20):
                node = (goal_pos * i/20) + (load_pos * (20-i)/20)
                node_x, node_y = int(node[0]), int(node[1])
                straight &= not obs_map[node_x, node_y]
            if straight and np.random.rand() < self.regenerate_ratio:
                return self.reset()
            
            self.path = []
            for i in range(self.astar_node-1):
                idx = int(len(self.astar_path)/self.astar_node*i)
                self.path.append(self.astar_path[idx][0])
                self.path.append(self.astar_path[idx][1])
            self.path.append(self.goal[0])
            self.path.append(self.goal[1])
            
            qpos = np.concatenate([qpos[:-self.astar_node*2-2], self.path, self.goal])
            self.set_state(np.array(qpos), np.zeros_like(qpos))
            
            done, _ = self._get_done(0)
            if done:
                return self.reset()
            
            # self.env_data[self.env_idx] = {
            #     'anchor': self.anchor_id.copy().tolist(),
            #     'qpos': qpos.copy().tolist(),
            #     'goal':self.goal.copy().tolist(),
            #     'path4render': self.path.copy(),
            #     'astar_path': self.astar_path.copy().tolist(),
            #     'init_obs_pos': self.init_obs_pos.copy().tolist(),
            #     'init_obs_yaw': self.init_obs_yaw.copy().tolist(),
            # }
            
            # import json
            # with open("results/all_data/{:04d}.json".format(self.env_idx), "w") as file:
            #     json.dump(self.env_data[self.env_idx], file)
            # print("current env idx", self.env_idx)
            # self.env_idx += 1
            
        
        # init variables
        self.t = 0.
        self.arrive_time = 0.
        self.total_rew = 0.
        load_pos = self.sim.data.qpos.copy()[:2]
        self.init_dist = np.linalg.norm(self.goal - load_pos)
        self.max_time = self.init_dist * 8. + 20. #self.init_dist * 8. + 20.
        self.hist_load = []
        # RL_info
        observation = self._get_obs() 
        info = dict()
        # post process
        self._post_update()
        
        return observation, info
    
    def _get_obs(self):
        
        # get current observation
        cur_vec_obs, cur_img_obs, cur_vec_sta = self._get_cur_obs()
        # update history
        self._update_history(cur_vec_obs, cur_img_obs)

        # reshape 
        vec_obs = cur_vec_obs.reshape([self.num_agent, -1]) # self.hist_vec_obs.reshape([self.num_agent, -1])        
        img_obs = cur_img_obs.reshape([self.num_agent, -1, *self.observation_space[1].shape[-2:]]) # self.hist_img_obs.reshape([self.num_agent, -1, *self.observation_space[1].shape[-2:]])
        
        # get local goal
        load_pos = self.sim.data.qpos.copy()[0:2]
        load_pos = np.expand_dims(load_pos[:2].copy(), 0)
        path_dist = np.linalg.norm(load_pos-self.astar_path, axis=-1)
        goal_idx = min(np.argmin(path_dist, axis=-1)+18, len(self.astar_path)-1) 
        local_goal = self.astar_path[goal_idx]
        local_goal = (self.goal - local_goal) / self.msize
        local_goal = np.expand_dims(local_goal, 0)
        local_goal = self._cart2polar(local_goal)
        
        # vector state
        vec_sta = cur_vec_sta[self.order].reshape([1, -1])
        if self.previledge_critic:
            kp = self.kp / self.init_kp
            kd = self.kd / self.init_kd
            time = (self.max_time-self.t) / 50.
            mass = self.load_mass / 5.
            vec_sta = np.concatenate([vec_sta, [[time]], kp, kd], -1)
            vec_sta = np.concatenate([vec_sta, local_goal, [[mass]]], -1)
            vec_sta = np.concatenate([vec_sta, [self.cable_len[self.order]]], axis=-1)  
        vec_sta = np.repeat(vec_sta, self.num_agent, axis=0)
        
        # add local goal to actor input
        local_goal = np.repeat(local_goal, self.num_agent, axis=0)
        vec_obs = np.concatenate([vec_obs, local_goal], axis=-1) 
        
        # shuffle
        vec_obs = vec_obs[self.order]
        img_obs = img_obs[self.order]
        img_sta = img_obs.copy()
        img_sta = img_sta.reshape([1, -1, *self.observation_space[1].shape[-2:]])
        img_sta = np.repeat(img_sta, self.num_agent, axis=0)
        
        return vec_obs, img_obs, vec_sta, img_sta

    def _update_history(self, cur_vec_obs, cur_img_obs):
        
        self.hist_vec_obs[:,:-1] = self.hist_vec_obs[:,1:]
        self.hist_img_obs[:,:-1] = self.hist_img_obs[:,1:]
        self.hist_vec_obs[:,-1] = cur_vec_obs.copy()
        self.hist_img_obs[:,-1] = cur_img_obs.copy()

    def step(self, cmd):
        """
        cmd(np.array): [num_agent, 3]
        observation(tuple): 
        reward(float)
        done(bool)
        info(dict)
        """
        # rearange actions
        idx = np.argsort(self.order)
        command = cmd[idx]
        # pre process
        command = np.clip(command, self.action_space.low, self.action_space.high)
        # action = command.copy()
        done, contact = self._do_simulation(command, self.frame_skip)
        self.t += self.dt
        # update RL info
        observation = self._get_obs()
        reward, info = self._get_reward(contact)
        self.total_rew += reward
        # post process
        self._post_update()
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
            if "obstacle" in obj1 or "obstacle" in \
                obj2 or "wall" in obj1 or "wall" in obj2:
                terminate, contact = True, True
            else:
                terminate, contact = True, False
            return terminate, contact

        # load contact rope done
        state = self.sim.data.qpos.copy().flatten()
        robot_state = state[4:4+self.num_agent*4]
        robot_state = robot_state.reshape([self.num_agent, 4])
        robot_pos = robot_state[:,:2]
        load_pos = state[:2].reshape([1, 2])
        load_yaw = state[2]
        anchor_yaw = self.anchor_id * np.pi/2 + load_yaw
        anchor_pos = np.stack([
            np.cos(anchor_yaw)*self.box_half_len+load_pos[:,0],
            np.sin(anchor_yaw)*self.box_half_len+load_pos[:,1],
        ], axis=-1)
        anchor2robot = robot_pos - anchor_pos
        anchor2robot_yaw = np.arctan2(anchor2robot[:,1], anchor2robot[:,0])
        cos_dist = np.cos(anchor2robot_yaw)*np.cos(anchor_yaw)
        cos_dist += np.sin(anchor2robot_yaw)*np.sin(anchor_yaw)
        if (cos_dist < np.cos(np.pi*3/4)).any():
            terminate, contact = True, False
            return terminate, contact
        
        # out of time
        if self.t > self.max_time:
            terminate, contact = True, False
            return terminate, contact
        
        # reach goal for a while
        state = self.sim.data.qpos.copy().flatten()
        dist = np.linalg.norm(state[:2]-self.goal)
        goal_reach = (dist<=self.arrive_dist)
        if self.arrive_time > self.arrive_thresh and goal_reach:
            terminate, contact = True, False
            return terminate, contact
        
        # load stuck for a while
        if self.t > 2 * self.stuck_time:
            dist = np.linalg.norm(state[:2]-self.hist_load[0])
            if dist < 0.2:
                terminate, contact = True, False
                return terminate, contact
            
        # if self.t > 5.:
        #     state = self.sim.data.qpos.copy().flatten()
        #     robot_state = state[4:4+self.num_agent*4]
        #     robot_state = robot_state.reshape([self.num_agent, 4])
        #     robot_pos = robot_state[:,:2]
        #     load_pos = state[:2].reshape([1, 2])
        #     load2robot = robot_pos - load_pos
        #     cosine_distance = np.dot(load2robot[0], load2robot[1])
        #     cosine_distance /= np.linalg.norm(load2robot[0])
        #     cosine_distance /= np.linalg.norm(load2robot[1])
        #     robot_yaw = robot_state[:,2]
        #     robot0_dir = np.array([np.cos(robot_yaw[0]), np.sin(robot_yaw[0])])
        #     robot1_dir = np.array([np.cos(robot_yaw[1]), np.sin(robot_yaw[1])])
        #     cosine_distance2 = np.dot(robot0_dir, robot1_dir)
        #     if cosine_distance < np.cos(np.pi*.75) and cosine_distance2 < np.cos(np.pi*.6) and not goal_reach:
        #         terminate, contact = True, False
        #         return terminate, contact
        
        return terminate, contact

    def _get_reward(self, contact):

        rewards = []
        # pre-process
        weights = np.array([1., 1., -0.001, 1., 0.])
        weights = weights / weights.sum()
        state = self.sim.data.qpos.copy().flatten()
        # # dist_per_step
        # local_dist = np.linalg.norm(state[:2]-self.last_local_goal)
        # dist = np.linalg.norm(state[:2]-self.goal)
        # rewards.append((self.prev_dist-local_dist)*(dist>self.arrive_dist))
        
        # dist_per_step
        local_dist = np.linalg.norm(state[:2]-self.last_local_goal)
        dist = np.linalg.norm(state[:2]-self.goal)
        dist_rew = (self.prev_dist - local_dist) * (dist > self.arrive_dist)
        # state = self.sim.data.qpos.copy().flatten()
        # robot_state = state[4:4+self.num_agent*4]
        # robot_state = robot_state.reshape([self.num_agent, 4])
        # robot_pos = robot_state[:,:2]
        # load_pos = state[:2].reshape([1, 2])
        # load2robot = robot_pos - load_pos
        # cosine_distance = np.dot(load2robot[0], load2robot[1])
        # cosine_distance /= np.linalg.norm(load2robot[0])
        # cosine_distance /= np.linalg.norm(load2robot[1])
        # dist_rew = dist_rew * (dist_rew<0) + \
        #     dist_rew * (dist_rew>0) * (cosine_distance>np.cos(np.pi*.75))
        rewards.append(dist_rew)
        # goal_reach
        max_speed = np.linalg.norm(self.action_space.high[:2])
        goal_reach = (dist<=self.arrive_dist) 
        rewards.append(goal_reach*(max_speed*self.dt+0.0006))
        # contact
        rewards.append(contact)
        # arrive for a while
        arrive4awhile = self.arrive_time > self.arrive_thresh
        max_rew = max_speed * self.dt * goal_reach + 0.0006
        remain_t = int((self.max_time - self.t) / self.dt)
        accu_coef = (1 - self.gamma**remain_t) / (1 - self.gamma)
        rewards.append(accu_coef * max_rew * arrive4awhile)
        
        state = self.sim.data.qpos.copy().flatten()
        robot_state = state[4:4+self.num_agent*4]
        robot_state = robot_state.reshape([self.num_agent, 4])
        robot_yaw = robot_state[:,2]
        load_pos = state[:2].copy().reshape([1,2])
        load2robot = robot_state[:,:2] - load_pos
        cosine, cos_cnt = 0., 0
        for i in range(self.num_agent):
            for j in range(i+1, self.num_agent):    
                cosine += np.cos(robot_yaw[i]-robot_yaw[j])
                cosine += np.dot(load2robot[i], load2robot[j]) / np.linalg.norm(load2robot[i]) / np.linalg.norm(load2robot[j])
                cos_cnt += 2
        cosine = cosine / cos_cnt
        dist = np.linalg.norm(state[:2]-self.goal)
        rewards.append((cosine-0.4) * (dist > self.arrive_dist))
        
        rew_dict = dict()
        rew_dict["dist_per_step"] = rewards[0]
        rew_dict["goal_reach"] = rewards[1]
        rew_dict["contact_done"] = rewards[2]
        rew_dict["arrive4awhile"] = rewards[3]
        rew_dict["cosine"] = rewards[4]
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
        load_p = self.sim.data.qpos.copy()[0:2]
        load_p += (np.random.random(load_p.shape)-.5) * self.measure_random_scale
        load_p = load_p.reshape([1,2])
        load_pos = (self.goal.reshape([1,2])-load_p) / self.msize
        load_pos = self._cart2polar(load_pos)
        load_pos = np.repeat(load_pos, self.num_agent, axis=0)
        load_y = self.sim.data.qpos.copy()[2:3]
        load_y += (np.random.random(load_y.shape)-.5) * self.measure_random_scale
        load_yaw = load_y.reshape([1,1])
        load_yaw = np.repeat(load_yaw, self.num_agent, axis=0)
        load_yaw = [np.sin(load_yaw), np.cos(load_yaw)]
        dog_state = self.sim.data.qpos.copy()[4:4+4*self.num_agent]
        dog_state = dog_state.reshape([self.num_agent, 4])
        dog_p = dog_state[:,0:2] 
        dog_p += (np.random.random(dog_p.shape)-.5) * self.measure_random_scale
        dog_pos = (self.goal.reshape([1,2])-dog_p) / self.msize
        dog_pos = self._cart2polar(dog_pos)
        dog_y = dog_state[:,2:3]
        dog_y += (np.random.random(dog_y.shape)-.5) * self.measure_random_scale
        dog_yaw = [np.sin(dog_y), np.cos(dog_y)]
        anchor_yaw = self.anchor_id * np.pi/2 + load_y[0]
        anchor_vec, _ = self._get_toward(anchor_yaw.reshape([self.num_agent, 1]))
        anchor_vec = anchor_vec * self.box_half_len + load_p
        anchor_vec = (self.goal.reshape([1,2])-anchor_vec) / self.msize
        anchor_vec = self._cart2polar(anchor_vec)
        cur_vec_obs = [load_pos, dog_pos] + load_yaw + dog_yaw + [anchor_vec] 
        cur_vec_obs = np.concatenate(cur_vec_obs, axis=-1)
        # print("cur_vec_obs.shape", cur_vec_obs.shape) # [num_agent, vec_shape]
        if self.previledge_critic:
            cur_vec_sta = [self.error, self.d_output, self.prev_output_vel]
            cur_vec_sta = np.concatenate(cur_vec_sta, axis=-1)
            cur_vec_sta = np.concatenate([cur_vec_obs, cur_vec_sta], axis=-1)
        else:
            cur_vec_sta = np.concatenate([cur_vec_obs], axis=-1)
        # image: partial observation
        cur_img_obs = []
        for i in range(self.num_agent):
            cur_img_o = []
            obs_pos = self.init_obs_pos.copy()
            obs_pos += (np.random.random(obs_pos.shape)-.5) * self.measure_random_scale
            obs_yaw = self.init_obs_yaw.copy()
            obs_yaw += (np.random.random(obs_yaw.shape)-.5) * self.measure_random_scale
            obs_len = np.ones([2])
            obs_map = self._draw_map(
                dog_p[i], dog_y[i][0], 
                obs_pos, obs_yaw, obs_len
            )
            wall_len = np.array([self.msize+2, 2.])
            obs_map += self._draw_map(
                dog_p[i], dog_y[i][0], 
                self.wall_pos, self.wall_yaw, wall_len
            )
            cur_img_o.append((obs_map!=0)*1.)
            
            # obs_map = np.zeros(self.observation_space[1].shape[1:]) # TODO
            # cur_img_o.append((obs_map!=0)*1.)
            
            load_pos = load_p.reshape([1,2])
            load_yaw = load_y.reshape([1,1])
            load_len = np.ones([2]) * self.box_half_len * 2.
            load_map = self._draw_map(
                dog_p[i], dog_y[i][0], 
                load_pos, load_yaw, load_len
            )
            cur_img_o.append((load_map!=0)*1.)
            
            all_dog_pos = dog_p.copy()
            all_dog_pos = np.delete(all_dog_pos, i, axis=0)
            all_dog_yaw = dog_y.copy()
            all_dog_yaw = np.delete(all_dog_yaw, i, axis=0)
            dog_len = np.array([0.75, 0.4])
            dog_map = self._draw_map(
                dog_p[i], dog_y[i][0], 
                all_dog_pos, all_dog_yaw, dog_len
            )
            cur_img_o.append((dog_map!=0)*1.)
            
            cur_img_o = np.stack(cur_img_o, axis=0)
            cur_img_obs.append(cur_img_o)
        cur_img_obs = np.stack(cur_img_obs, axis=0)
        # import imageio
        # imageio.imwrite("../envs/mujoco/assets/box.png", load_map)
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
        # n_frames = n_frames * (1. + (np.random.rand()-.5) * self.domain_random_scale)
        for _ in range(int(n_frames)):
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
        # self.intergral += error * self.ki * dt_line_segments_intersect
        self.d_output = cur_vel - self.prev_output_vel
        torque = self.kp * self.error - self.kd * (self.d_output/dt) # + self.intergral
        torque_low = np.repeat(np.array([[-13*9.83,-10.,-100]]), self.num_agent, 0)
        torque_high = np.repeat(np.array([[13*9.83,10.,100]]), self.num_agent, 0)
        torque = np.clip(torque, torque_low, torque_high)
        self.prev_output_vel = cur_vel.copy()
        return torque.flatten()

    def _post_update(self):
        """
        restore the history information
        """
        # update last distance
        load_pos = self.sim.data.qpos.copy()[0:2]
        load_pos = np.expand_dims(load_pos[:2].copy(), 0)
        path_dist = np.linalg.norm(load_pos-self.astar_path, axis=-1)
        goal_idx = min(np.argmin(path_dist, axis=-1)+18, len(self.astar_path)-1) 
        local_goal = self.astar_path[goal_idx]
        self.prev_dist = np.linalg.norm(load_pos-local_goal)
        self.last_local_goal = local_goal.copy()
        
        qpos = np.concatenate([self.sim.data.qpos.copy()[:-2], self.last_local_goal])
        qvel = self.sim.data.qvel.copy()
        self.set_state(np.array(qpos), qvel)
        self.total_step += 1
        
        load_pos = self.sim.data.qpos.copy()[0:2]
        dist = np.linalg.norm(load_pos-self.goal)
        if dist < self.arrive_dist:
            self.arrive_time += self.dt
        else:
            self.arrive_time = 0.
        
        self.hist_load.append(load_pos.copy())
        if len(self.hist_load) > self.stuck_time/self.dt:
            self.hist_load.pop(0)
        
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
