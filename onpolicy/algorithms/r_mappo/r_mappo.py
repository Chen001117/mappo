import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check

class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, policy, device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta    
        self.num_agents = args.num_agents

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = []
            for _ in range(self.num_agents):
                value_norm = ValueNorm(1).to(self.device)
                self.value_normalizer.append(value_norm)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch, num_agents):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        if self._use_popart or self._use_valuenorm:
            error_clipped = torch.zeros_like(return_batch)
            error_original = torch.zeros_like(return_batch)
            return4update = []
            for i in range(self.num_agents):
                agent_id = num_agents[:,0] == (i+1)
                active_id = active_masks_batch[:,0][agent_id] == 1.
                return4update.append(return_batch[agent_id][active_id])
            return4update = torch.cat(return4update, dim=0)
            self.value_normalizer[i].update(return4update)
            for i in range(self.num_agents):
                agent_id = num_agents[:,0] == (i+1)
                error_clipped[agent_id] = self.value_normalizer[i].normalize(return_batch[agent_id]) - value_pred_clipped[agent_id]
                error_original[agent_id] = self.value_normalizer[i].normalize(return_batch[agent_id]) - values[agent_id]
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        # value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, num_agents, num_agents_rnn = sample

        if active_masks_batch.sum() == 0:
            print("skip this batch, due to the sum of active masks is zeros.")
            return torch.zeros(1), 0., torch.zeros(1), torch.zeros(1), 0., 1.

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch, 
            obs_batch, 
            rnn_states_batch, 
            rnn_states_critic_batch, 
            actions_batch, 
            masks_batch, 
            num_agents,
            num_agents_rnn,
            available_actions_batch,
            active_masks_batch,
        )
        # active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        ratio = (imp_weights * active_masks_batch).sum() / active_masks_batch.sum() / imp_weights.shape[-1]
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        
        policy_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) \
                    * active_masks_batch).sum() / active_masks_batch.sum()

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch, num_agents)

        for critic_optimizer in self.policy.critic_optimizer:
            critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        for critic in self.policy.critic:
            if self._use_max_grad_norm:
                critic_grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
            else:
                critic_grad_norm = get_gard_norm(critic.parameters())

        for critic_optimizer in self.policy.critic_optimizer:
            critic_optimizer.step()
        
        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, ratio

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        
        low_obs_vec = buffer.share_obs_vec[:-1].copy()
        low_obs_img = buffer.share_obs_img[:-1].copy() # [len, n_env, n_agent, c, w ,h] 
        
        e_len, n_env, n_agent, _ = low_obs_vec.shape
        
        cur_vec_sta = low_obs_vec[:,:,:,0:15] # [len, n_env, n_agent, n_size] 
        hist_vec_sta = low_obs_vec[:,:,:,15:78] # [len, n_env, n_agent, n_size] 
        anchor_vec = low_obs_vec[:,:,:,78:86] # [len, n_env, n_agent, n_size] 
        agent_id = low_obs_vec[:,:,:,86:88] # [len, n_env, n_agent, n_size] 
        time = low_obs_vec[:,:,:,88:89] # [len, n_env, n_agent, n_size] 
        
        def get_01(obs):
            o0 =  np.concatenate([
                obs[:,:,:,0:3], obs[:,:,:,3:6], 
                obs[:,:,:,9:11], obs[:,:,:,11:13]
            ], axis=-1)
            o1 =  np.concatenate([
                obs[:,:,:,0:3], obs[:,:,:,6:9], 
                obs[:,:,:,9:11], obs[:,:,:,13:15]
            ], axis=-1)
            return o0, o1
    
        def get_hist_01(obs):
            obs = obs.reshape([e_len, n_env, n_agent, 3, 21])
            o0s, o1s = [], []
            for i in range(3):
                o0, o1 = get_01(obs[:,:,:,i,:15])
                c0 = obs[:,:,:,i,15:18]
                c1 = obs[:,:,:,i,18:21]
                o0s.append(o0)
                o0s.append(c0)
                o1s.append(o1)
                o1s.append(c1)
            return np.concatenate(o0s,-1), np.concatenate(o1s, -1)
        
        cur_vec_sta_0, cur_vec_sta_1 = get_01(cur_vec_sta)
        hist_vec_sta_0, hist_vec_sta_1 = get_hist_01(hist_vec_sta)
        anchor_vec_0, anchor_vec_1 = anchor_vec[:,:,:,:4], anchor_vec[:,:,:,4:]
        agent_id_0, agent_id_1 = np.ones([e_len, n_env, n_agent, 1]), np.ones([e_len, n_env, n_agent, 1])
        sta_0 = np.concatenate([
            cur_vec_sta_0, hist_vec_sta_0, anchor_vec_0, agent_id_0, time
        ], axis=-1)
        sta_1 = np.concatenate([
            cur_vec_sta_1, hist_vec_sta_1, anchor_vec_1, agent_id_1, time
        ], axis=-1)
        low_obs_vec = np.concatenate([sta_1[:,:,:1], sta_0[:,:,:1]], 2)
        # print("OBS", low_obs_vec.shape)
        
        low_obs_img_0 = np.concatenate([
            low_obs_img[:,:,:,0:2], low_obs_img[:,:,:,2:4], 
            low_obs_img[:,:,:,6:8], low_obs_img[:,:,:,8:10],
            low_obs_img[:,:,:,12:14], low_obs_img[:,:,:,14:16],
            low_obs_img[:,:,:,18:20], low_obs_img[:,:,:,20:22],
        ], axis=3)
        for i in range(8):
            low_obs_img_0[:,:,:,i*2+1] *= 0.
        low_obs_img_1 = np.concatenate([
            low_obs_img[:,:,:,0:2], low_obs_img[:,:,:,4:6], 
            low_obs_img[:,:,:,6:8], low_obs_img[:,:,:,10:12],
            low_obs_img[:,:,:,12:14], low_obs_img[:,:,:,16:18],
            low_obs_img[:,:,:,18:20], low_obs_img[:,:,:,22:24],
        ], axis=3)
        for i in range(8):
            low_obs_img_1[:,:,:,i*2+1] *= 0.
        low_obs_img = np.concatenate([low_obs_img_1[:,:,:1], low_obs_img_0[:,:,:1]], 2)
        # print("IMG", low_obs_img.shape)

        rnn = np.concatenate(buffer.rnn_states_critic[0]) 
        advantages = []
        for i in range(e_len):
            obs = (
                np.concatenate(low_obs_vec[i], 0),
                np.concatenate(low_obs_img[i], 0),
            )
            mask = np.concatenate(buffer.masks[i,:,:,:1].copy(), 0)
            rnn[mask[:,0]==True] = np.zeros([(mask==True).sum(), 1, 256])
            value, rnn = self.policy.critic[0](obs, rnn, mask)
            rnn = rnn.cpu().detach().numpy()
            value = self.value_normalizer[0].denormalize(value.cpu().detach().numpy())
            adv = buffer.returns[i].copy() - value.reshape([n_env, n_agent, 1])
            advantages.append(adv)
        colab_advs = np.array(advantages)
        # place_holder_masks = (buffer.share_obs_vec[:-1].copy()==0).all(axis=-1)
        single_agent_masks = (buffer.num_agents[:-1].copy()==1)
        colab_advs[single_agent_masks] = 0.
        # colab_advs[place_holder_masks] = 0.
        colab_advs = np.clip(np.exp(colab_advs), 0., 2.)
        
        train_info = {}
        if self._use_popart or self._use_valuenorm:
            advantages = np.zeros_like(np.concatenate(buffer.returns[:-1], axis=0))
            for i in range(self.num_agents):
                agent_id = buffer.num_agents[:-1,:,0,0] == (i+1)
                agent_id = np.concatenate(agent_id, axis=0)
                returns = np.concatenate(buffer.returns[:-1].copy(), axis=0)[agent_id]
                values = np.concatenate(buffer.value_preds[:-1].copy(), axis=0)[agent_id]
                advantages[agent_id] = (returns - self.value_normalizer[i].denormalize(values))
            advantages = advantages.reshape(buffer.returns[:-1].shape)
            advantages *= colab_advs
            train_info['colab_advs'] = colab_advs.mean()
        # else:
        #     advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks.copy()[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, ratio \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += ratio

        num_updates = self.ppo_epoch * self.num_mini_batch
        train_info['colab_advs'] *= num_updates

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        for critic in self.policy.critic:
            critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        for critic in self.policy.critic:
            critic.eval()