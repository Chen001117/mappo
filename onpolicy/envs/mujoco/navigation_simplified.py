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

        # hyper params:
        self._reach_threshold = 0.5 # m
        self._num_agents = num_agents # number of agents
        self._num_obstacles = 40 # number of obstacles
        self._domain_random_scale = 1e-1 # domain randomization scale
        self._measure_random_scale = 1e-2 # measurement randomization scale
        self._num_frame_skip = 100 # 1000/frame_skip = decision_freq (Hz)
        self._map_real_size = 20. # map size (m)
        self._warm_step = 2 # warm-up: let everything stable (steps)
        self._num_astar_nodes = 20 # number of nodes for rendering astar path

        # simulator params
        self._init_kp = np.array([[2000, 2000, 800]]) # kp for PD controller
        self._init_kd = np.array([[0.02, 0.02, 0.0]]) # kd for PD controller
        self._robot_size = np.array([0.65, 0.3, 0.3]) # m
        self._obstacle_size = np.array([1., 1., 1.]) # m
        self._load_size = np.array([0.6, 0.6, 0.6]) # m

        # initialize simulator
        self._anchor_id = np.random.randint(0, 4, self._num_agents)
        init_load_mass = 3. # kg
        self._load_mass = init_load_mass * (1 + self._rand(self._domain_random_scale))
        init_cable_len = np.ones(self._num_agents) # m
        self._cable_len = init_cable_len * (1 + self._rand(self._domain_random_scale))
        init_fric_coef = 1. # friction coefficient
        self._fric_coef = init_fric_coef * (1 + self._rand(self._domain_random_scale))
        self._astar_node = 3 # for rendering astar path
        model = get_xml(
            dog_num = self._num_agents, 
            obs_num = self._num_obstacles, 
            anchor_id = self._anchor_id,
            load_mass = self._load_mass,
            cable_len = self._cable_len,
            fric_coef = self._fric_coef,
            astar_node = self._num_astar_nodes
        )

        # RL space
        local_observation_size = 3+self._num_obstacles*3+3+3
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(local_observation_size,), dtype=np.float64
        )
        global_state_size = self._num_agents*3+self._num_obstacles*3+3+3
        self.share_observation_space = Box(
            low=-np.inf, high=np.inf, shape=(global_state_size,), dtype=np.float64
        )
        self.action_space = Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
        )

    def reset(self):
        # init task specific parameters
        self._kp  = self._init_kp * (1 + self._rand(self._domain_random_scale, 3))
        self._kd  = self._init_kd * (1 + self._rand(self._domain_random_scale, 3))
        # init variables
        self._t = 0.
        self._max_time = 1e6
        # reset simulator
        self._reset_simulator()
        # get rl info
        observation = self._get_obs()
        info = dict()
        # post process
        self._post_update()
        return observation, info
    
    def step(self, action):
        # step simulator
        action = np.clip(action, self.action_space.low, self.action_space.high)
        done = self._do_simulation(action, self._num_frame_skip)
        # get rl info
        observation = self._get_obs()
        reward, info = self._get_reward()
        # post process
        self._post_update()
        return observation, reward, done, info
        
    def _reset_simulator(self):
        # initialize obstacles
        init_obstacles_pos = self._rand(self._map_real_size, [self._num_obstacles, 2])
        init_obstacles_yaw = self._rand(2*np.pi, [self._num_obstacles, 1])
        init_obstacles_z = np.ones([self._num_obstacles, 1]) * self._obstacle_size[2]/2
        # initialize load
        init_load_pos = self._rand(self._map_real_size, [1, 2])
        init_load_yaw = self._rand(2*np.pi, [1, 1])
        init_load_z = np.ones([1, 1]) * self._load_size[2]/2
        # initialize dogs
        init_anchor_robot_len = self._rand(0.5, [self._num_agents, 1]) + 0.75
        init_anchor_robot_yaw = self._rand(np.pi, [self._num_agents, 1])
        init_anchor_id = self._anchor_id.reshape([self._num_agents, 1])
        init_anchor_pos = self._get_toward(init_load_yaw)[0] * 0.3 * (init_anchor_id==0)
        init_anchor_pos += self._get_toward(init_load_yaw)[1] * 0.3 * (init_anchor_id==1)
        init_anchor_pos += self._get_toward(init_load_yaw)[0] * (-0.3) * (init_anchor_id==2)
        init_anchor_pos += self._get_toward(init_load_yaw)[1] * (-0.3) * (init_anchor_id==3)
        init_load_robot_yaw = init_load_yaw + init_anchor_id * np.pi/2 + init_anchor_robot_yaw
        init_anchor_robot_pos = self._get_toward(init_load_robot_yaw)[0] * init_anchor_robot_len
        init_robot_pos = init_anchor_pos + init_anchor_robot_pos
        init_robot_yaw = self._rand(2*np.pi, [self._num_agents, 1])
        init_robot_z = np.ones([self._num_agents, 1]) * self._robot_size[2]/2
        # initialize goal
        self._goal = self._rand(self._map_real_size, [1, 2])
        # astar search
        astar_path = self._astar_search(init_load_pos, self._goal, init_obstacles_pos)
        if astar_path is None:
            return self._reset_simulator()
        # set state
        qpos = self.sim.data.qpos.copy()
        load = np.concatenate([init_load_pos, init_load_yaw, init_load_z], axis=-1).flatten()
        robots = np.concatenate([init_robot_pos, init_robot_yaw, init_robot_z], axis=-1).flatten()
        obstacles = np.concatenate([init_obstacles_pos, init_obstacles_yaw, init_obstacles_z], axis=-1).flatten()
        goal = self._goal.flatten()
        qpos[:-self._num_astar_nodes*2-12] = np.concatenate([load, robots, obstacles, goal]) 
        qpos[-self._num_astar_nodes*2:] = astar_path.flatten()
        self.set_state(qpos, np.zeros_like(qpos))
        # warm-up
        for _ in range(self._warm_step):
            terminated = self._do_simulation(0., self._num_frame_skip, add_time=False)
            if terminated:
                return self._reset_simulator()
    
    def _do_simulation(self, action, num_frame_skip, add_time=True):
        for _ in range(num_frame_skip):
            terminated = self._get_done()
            if terminated:
                return True
            robot_global_velocity = self._get_state("robot", vel=True)
            local_vel = self._global2local(robot_global_velocity)
            torque = self._pd_controller(target=action, cur=local_vel)
            self.sim.data.ctrl[:] = self._local2global(torque).flatten()
            self.sim.step()
            self._t = self._t + self.dt if add_time else self._t
        terminated = self._get_done()
        return terminated

    def _get_obs(self):
        # get load info
        load_state = self._get_state("load")
        load_pos = load_state[:,:2] - self._goal
        load_pos += self._rand(self._measure_random_scale, load_pos.shape)
        load_pos /= self._map_real_size
        load_pos = self._cart2polar(load_pos)
        load_yaw = load_state[:,2:3]
        load_yaw += self._rand(self._measure_random_scale, load_yaw.shape)
        load = np.concatenate([np.cos(load_yaw), np.sin(load_yaw)], axis=-1)
        # get robot info
        robot_state = self._get_state("robot")
        robot_pos = robot_state[:,:2] - self._goal
        robot_pos += self._rand(self._measure_random_scale, robot_pos.shape)
        robot_pos /= self._map_real_size
        robot_pos = self._cart2polar(robot_pos)
        robot_yaw = robot_state[:,2:3]
        robot_yaw += self._rand(self._measure_random_scale, robot_yaw.shape)
        robot= np.concatenate([np.cos(robot_yaw), np.sin(robot_yaw)], axis=-1)
        # get obstacle info
        obstacle_state = self._get_state("obstacle")
        obstacle_pos = obstacle_state[:,:2] - self._goal
        obstacle_pos += self._rand(self._measure_random_scale, obstacle_pos.shape)
        obstacle_pos /= self._map_real_size
        obstacle_pos = self._cart2polar(obstacle_pos)
        obstacle_yaw = obstacle_state[:,2:3]
        obstacle_yaw += self._rand(self._measure_random_scale, obstacle_yaw.shape)
        obstacle = np.concatenate([np.cos(obstacle_yaw), np.sin(obstacle_yaw)], axis=-1)
        # get local observation
        local_observation = np.concatenate([load, robot, obstacle], axis=-1)
        # get global state
        robot_state = robot.reshape([1, -1])
        robot_state = np.repeat(robot_state, self._num_agents, axis=0)
        global_state = np.concatenate([load, robot_state, obstacle], axis=-1)

        return local_observation, global_state
    
    def _get_reward(self):
        # weights
        weight = np.array([1., 1.])
        weight = weight / weight.sum()
        # get rewards
        rewards = []
        # dense reward
        load_pos = self._get_state("load")[:,:2]
        load_dist = np.linalg.norm(load_pos-self._goal)
        rewards.append(self.last_load_dist-load_dist)
        # sparse reward
        rewards.append(load_dist < self._reach_threshold)
        # calculate reward
        rewards = np.array(rewards)
        rewards = np.dot(weight, rewards)
        return rewards

    def _get_done(self):
        # contact done
        for i in range(self.sim.data.ncon):
            con = self.sim.data.contact[i]
            obj1 = self.sim.model.geom_id2name(con.geom1)
            obj2 = self.sim.model.geom_id2name(con.geom2)
            if obj1=="floor" or obj2=="floor":
                continue
            else:
                return True
        # out of time
        if self._t > self._max_time:
            return True
        return False

    def _post_update(self):
        load_pos = self._get_state("load")[:,:2]
        self.last_load_dist = np.linalg.norm(load_pos-self._goal)

    # return x ~ U[-0.5*scale, 0.5*scale]
    def _rand(self, scale, size=None):
        if size is None:
            return (np.random.random()-0.5)*scale
        else:
            return (np.random.random(size)-0.5)*scale

    def _cart2polar(self, x):
        r = np.linalg.norm(x, axis=-1, keepdims=True)
        theta = np.arctan2(x[:,1:2], x[:,0:1])
        cos, sin = np.cos(theta), np.sin(theta)
        return np.concatenate([r, cos, sin], axis=-1)

    def _get_state(self, data_name, vel=False):

        if vel:
            state = self.sim.data.qvel.copy()
        else:
            state = self.sim.data.qpos.copy()

        if data_name == 'load':
            state = state[:4]
        elif data_name == 'robot':
            state = state[4:4+self._num_agents*4]
        elif data_name == 'obstacle':
            state = state[4+self._num_agents*4:4+self._num_agents*4+self._num_obstacles*4]
        else:
            raise NotImplementedError
        
        state = state.reshape([self._num_agents, 4])

        return state[:,:3]

    def _get_toward(self, theta):
        forwards = np.concatenate([np.cos(theta), np.sin(theta)], axis=-1)
        verticals = np.concatenate([-np.sin(theta), np.cos(theta)], axis=-1)
        return forwards, verticals

    def _local2global(self, actions):
        actions = actions.reshape([self._num_agents, 3])
        robot_yaw = self._get_state("robot")[:,2:3]
        forwards, verticals = self._get_toward(robot_yaw)
        actions_linear = actions[:,0:1] * forwards + actions[:,1:2] * verticals 
        actions_angular = actions[:,2:3]
        actions = np.concatenate([actions_linear, actions_angular], axis=-1)
        return actions

    def _global2local(self, actions):
        actions = actions.reshape([self._num_agents, 3])
        actions_linear = actions[:,:2]
        actions_angular = actions[:,2:3]
        robot_yaw = self._get_state("robot")[:,2:3]
        rot_mat = np.array([[np.cos(robot_yaw), np.sin(robot_yaw)],[-np.sin(robot_yaw), np.cos(robot_yaw)]])
        rot_mat = np.transpose(rot_mat, [2,0,1])
        actions_linear = np.expand_dims(actions_linear, 2)
        actions_linear = rot_mat @ actions_linear
        actions_linear = actions_linear.squeeze(2) 
        actions = np.concatenate([actions_linear, actions_angular], -1)
        return actions

    def _pd_controller(self, target, cur_vel):
        self._error_vel = target - cur_vel
        self._d_output_vel = cur_vel - self._prev_output_vel
        torque = self._kp * self._error_vel 
        torque += self._kd * self._d_output_vel/self.model.opt.timestep
        torque = np.clip(torque, self._torque_low, self._torque_high)
        self._prev_output_vel = cur_vel.copy()
        return torque

