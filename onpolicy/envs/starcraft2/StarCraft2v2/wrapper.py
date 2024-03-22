from .distributions import get_distribution
from .starcraft2 import StarCraft2Env
from ..multiagentenv import MultiAgentEnv

import json
import numpy as np

class StarCraftCapabilityEnvWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.distribution_config = kwargs["capability_config"]
        self.env_key_to_distribution_map = {}
        self._parse_distribution_config()
        self.env = StarCraft2Env(**kwargs)
        assert (
            self.distribution_config.keys()
            == kwargs["capability_config"].keys()
        ), "Must give distribution config and capability config the same keys"
        
        # load reset config
        with open("config.json", "r") as f:
            data = json.load(f)
        self.all_config = data

        self.env_info = None
        self.win_cnt = np.zeros(800)
        self.all_cnt = np.zeros(800)
        self.idx = -1

    def _parse_distribution_config(self):
        for env_key, config in self.distribution_config.items():
            if env_key == "n_units" or env_key == "n_enemies":
                continue
            config["env_key"] = env_key
            # add n_units key
            config["n_units"] = self.distribution_config["n_units"]
            config["n_enemies"] = self.distribution_config["n_enemies"]
            distribution = get_distribution(config["dist_type"])(config)
            self.env_key_to_distribution_map[env_key] = distribution

    def reset(self):
        # reset_config = {}
        # for distribution in self.env_key_to_distribution_map.values():
        #     reset_config = {**reset_config, **distribution.generate()}
        
        if self.env_info is not None:
            if "battle_won" in self.env_info:
                won = 1. if self.env_info["battle_won"] else 0.
                self.win_cnt[self.idx] += won
                self.all_cnt[self.idx] += 1
            print("win rate", self.win_cnt.sum()/self.all_cnt.sum(), "total", self.all_cnt.sum())
            result = np.stack([self.win_cnt, self.all_cnt], 0)
            np.save("result3.npy", result)
        
        if self.all_cnt.sum() > 3200:
            exit()

        # self.idx = np.random.randint(800)
        self.idx = (self.idx+1) % 800
        reset_config = self.all_config[str(self.idx)]
        reset_config["ally_start_positions"]["item"] = np.array(reset_config["ally_start_positions"]["item"])
        reset_config["enemy_start_positions"]["item"] = np.array(reset_config["enemy_start_positions"]["item"])

        return self.env.reset(reset_config)

    def __getattr__(self, name):
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            raise AttributeError

    def get_obs(self):
        return self.env.get_obs()

    def get_obs_feature_names(self):
        return self.env.get_obs_feature_names()

    def get_state(self):
        return self.env.get_state()

    def get_state_feature_names(self):
        return self.env.get_state_feature_names()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_state_size(self):
        return self.env.get_state_size()

    def get_total_actions(self):
        return self.env.get_total_actions()

    def get_capabilities(self):
        return self.env.get_capabilities()

    def get_obs_agent(self, agent_id):
        return self.env.get_obs_agent(agent_id)

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def render(self):
        return self.env.render()

    def step(self, actions):
        reward, terminated, info = self.env.step(actions)
        self.env_info = info
        return reward, terminated, info

    def get_stats(self):
        return self.env.get_stats()

    def full_restart(self):
        return self.env.full_restart()

    def save_replay(self):
        self.env.save_replay()

    def close(self):
        return self.env.close()