import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule
from onpolicy.algorithms.utils.util import check

import numpy as np
from gym.spaces import Box, Tuple

class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.num_agents = args.num_agents
        self.data_chunk_length = args.data_chunk_length

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr, eps=self.opti_eps,
            weight_decay=self.weight_decay
        )
        
        self.critic = []
        self.critic_optimizer = []
        for n_agent in range(1,self.num_agents+1):
            hist_vec_sta_size = (5 + 8*n_agent) * 3
            sta_size = 5 + 5*n_agent + hist_vec_sta_size + n_agent + 4*n_agent + 1
            share_obs_space = Tuple((
                Box(low=-np.inf, high=np.inf, shape=(sta_size,), dtype=np.float64), 
                Box(low=-np.inf, high=np.inf, shape=(n_agent*8+8,57,57), dtype=np.float64),
            ))
            critic = R_Critic(args, share_obs_space, self.device)
            critic_optimizer = torch.optim.Adam(
                critic.parameters(),
                lr=self.critic_lr, eps=self.opti_eps,
                weight_decay=self.weight_decay
            )
            self.critic.append(critic)
            self.critic_optimizer.append(critic_optimizer)
            
        self.sta_vec_size = []
        for n_agent in range(1, self.num_agents+1):
            hist_vec_sta_size = 5 + 8*n_agent
            sta_size = 5 + 5*n_agent + hist_vec_sta_size*3 + n_agent + 4*n_agent + 1
            self.sta_vec_size.append(sta_size)
        self.sta_img_size = []
        for n_agent in range(1, self.num_agents+1):
            self.sta_img_size.append(n_agent*8+8)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        for i in range(self.num_agents):
            update_linear_schedule(self.critic_optimizer[i], episode, episodes, self.critic_lr)

    def get_actions(
        self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, num_agents,
            available_actions=None, deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        
        values = np.zeros([*actions.shape[:-1], 1])
        rnns = np.zeros_like(rnn_states_critic)
        for i in range(self.num_agents):
            agent_id = num_agents[:,0]==(i+1)
            vec_sta = cent_obs[0][agent_id][:,:self.sta_vec_size[i]]
            img_sta = cent_obs[1][agent_id][:,:self.sta_img_size[i]]
            val, rnn = self.critic[i](
                (vec_sta, img_sta), 
                rnn_states_critic[agent_id], 
                masks[agent_id]
            )
            values[agent_id] = val
            rnns[agent_id] = rnn
        
        return values, actions, action_log_probs, rnn_states_actor, rnns

    def get_values(self, cent_obs, rnn_states_critic, masks, num_agents):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        
        values = np.zeros([*num_agents.shape[:-1], 1])
        rnns = np.zeros_like(rnn_states_critic)
        for i in range(self.num_agents):
            agent_id = num_agents[:,0]==(i+1)
            vec_sta = cent_obs[0][agent_id][:,:self.sta_vec_size[i]]
            img_sta = cent_obs[1][agent_id][:,:self.sta_img_size[i]]
            val, rnn = self.critic[i](
                (vec_sta, img_sta), 
                rnn_states_critic[agent_id], 
                masks[agent_id]
            )
            values[agent_id] = val
            rnns[agent_id] = rnn
        
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, num_agents, num_agents_rnn,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        active_masks = (obs[0]!=0).any(-1) & (obs[1]!=0).any(-1).any(-1).any(-1)
        active_masks = active_masks.reshape([-1, 1])
        
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )
        
        values = torch.zeros([*num_agents.shape[:-1], 1])
        for i in range(self.num_agents):
            agent_id = num_agents[:,0]==(i+1)
            agent_id_rnn = num_agents_rnn[:,0]==(i+1)
            vec_sta = cent_obs[0][agent_id][:,:self.sta_vec_size[i]]
            img_sta = cent_obs[1][agent_id][:,:self.sta_img_size[i]]
            val, rnn = self.critic[i](
                (vec_sta, img_sta), 
                rnn_states_critic[agent_id_rnn], 
                masks[agent_id]
            )
            values[agent_id] = val
        values[~active_masks[:,0]] = 0.
        
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

    def get_value(self, cent_obs, rnn_states_critic, masks):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, rnn_states_critic