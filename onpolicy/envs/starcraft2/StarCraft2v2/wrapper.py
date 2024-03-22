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

        # self.env_info = None
        # self.win_cnt = np.zeros(800)
        # self.all_cnt = np.zeros(800)

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
        
        # if self.env_info is not None:
        #     if "battle_won" in self.env_info:
        #         won = 1. if self.env_info["battle_won"] else 0.
        #         self.win_cnt[self.idx] += won
        #         self.all_cnt[self.idx] += 1
        #     print("win rate", self.win_cnt.sum()/self.all_cnt.sum(), "total", self.all_cnt.sum())
        #     result = np.stack([self.win_cnt, self.all_cnt], 0)
        #     np.save("result.npy", result)
        
        solved_task = np.array([  
            1,   3,   6,   7,   8,  14,  19,  22,  23,  24,  26,  28,  30,
            34,  35,  40,  41,  42,  43,  47,  50,  52,  56,  63,  64,  67,
            71,  75,  84,  91,  92,  97,  98,  99, 103, 105, 109, 116, 118,
            119, 120, 121, 125, 127, 129, 130, 133, 134, 135, 139, 147, 149,
            152, 153, 154, 157, 161, 163, 165, 171, 179, 184, 185, 186, 189,
            190, 193, 197, 198, 201, 204, 209, 211, 213, 215, 216, 217, 219,
            225, 232, 236, 238, 241, 244, 249, 252, 253, 260, 262, 263, 265,
            266, 269, 271, 278, 279, 280, 284, 285, 287, 288, 297, 299, 300,
            301, 304, 306, 307, 309, 318, 319, 320, 324, 325, 326, 333, 335,
            340, 341, 344, 347, 349, 350, 351, 354, 356, 358, 360, 364, 373,
            378, 381, 398, 399, 407, 413, 417, 418, 419, 421, 426, 430, 431,
            433, 434, 437, 442, 444, 445, 447, 449, 454, 456, 457, 460, 463,
            466, 467, 471, 473, 475, 477, 481, 485, 486, 496, 500, 501, 504,
            506, 510, 513, 517, 518, 525, 529, 536, 540, 541, 542, 543, 544,
            546, 547, 552, 555, 558, 560, 562, 564, 567, 570, 571, 575, 587,
            588, 590, 595, 598, 600, 601, 607, 609, 612, 619, 620, 623, 635,
            638, 639, 642, 643, 644, 649, 650, 655, 660, 663, 665, 666, 668,
            669, 670, 671, 672, 673, 674, 675, 681, 686, 687, 693, 695, 697,
            699, 700, 701, 702, 709, 710, 717, 723, 730, 731, 734, 737, 741,
            742, 750, 753, 756, 762, 763, 769, 770, 774, 775, 784, 786, 789,
            795, 796, 798, 799, 
            4,  17,  21,  25,  33,  39,  68,  85,  87,  89,  94,  96, 100,
            111, 123, 137, 138, 141, 142, 143, 145, 162, 168, 177, 180, 192,
            194, 212, 214, 218, 221, 223, 224, 227, 228, 234, 239, 247, 254,
            270, 272, 275, 291, 293, 294, 296, 302, 303, 327, 338, 348, 355,
            357, 361, 363, 366, 369, 374, 377, 380, 389, 397, 402, 403, 412,
            423, 429, 439, 443, 446, 453, 455, 464, 468, 472, 478, 479, 484,
            499, 508, 515, 516, 522, 524, 526, 531, 532, 556, 581, 584, 593,
            594, 597, 602, 603, 604, 606, 610, 611, 616, 617, 625, 629, 632,
            633, 634, 640, 641, 647, 653, 654, 656, 661, 662, 664, 676, 677,
            679, 694, 696, 705, 707, 708, 724, 727, 733, 736, 739, 744, 755,
            757, 759, 760, 764, 780, 783,
            0,   5,  11,  12,  27,  32,  37,  51,  54,  72,  74,  77,  82,
            86,  88,  93,  95, 112, 117, 124, 126, 128, 132, 136, 148, 159,
            167, 183, 187, 188, 196, 207, 220, 222, 235, 245, 248, 261, 267,
            273, 286, 289, 305, 316, 321, 322, 331, 332, 337, 345, 359, 365,
            370, 382, 384, 390, 392, 394, 400, 404, 406, 410, 414, 416, 432,
            480, 482, 489, 492, 494, 497, 505, 507, 509, 519, 528, 530, 534,
            539, 548, 549, 550, 553, 554, 559, 561, 565, 568, 574, 578, 579,
            589, 591, 605, 608, 615, 621, 646, 657, 658, 667, 680, 688, 689,
            692, 704, 711, 712, 716, 732, 749, 765, 767, 771, 778, 782, 785,
            788, 794
        ])

        self.idx = np.random.randint(800)
        while self.idx in solved_task:
            self.idx = np.random.randint(800)
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
        # self.env_info = info
        return reward, terminated, info

    def get_stats(self):
        return self.env.get_stats()

    def full_restart(self):
        return self.env.full_restart()

    def save_replay(self):
        self.env.save_replay()

    def close(self):
        return self.env.close()