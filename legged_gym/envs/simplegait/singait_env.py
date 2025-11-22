from legged_gym.envs.simplegait.base_gait_env import BaseGaitEnv
from isaacgym.torch_utils import torch_rand_float

import torch
# def _hugwbc_polynomial_planer(t0, t1, x0, x1, v0=0, v1=0, a0=0, a1=0):

#     T = t1 - t0
#     h = x1 - x0
#     k0 = x0
#     k1 = v0
#     k2 = 0.5 * a0
#     k3 = (20 * h - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * (T ** 2)) / (2 * (T ** 3))
#     k4 = (-30 * h + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * (T ** 2)) / (2 * (T ** 4))
#     k5 = (12 * h - 6 * (v1 + v0) * T + (a1 - a0) * (T ** 2)) / (2* (T ** 5))
#     coef = [k0, k1, k2, k3, k4, k5]

#     return coef


class SinGaitEnv(BaseGaitEnv):
    
    def _init_custom_buffers(self):
        super()._init_custom_buffers()
        
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, 
                                                requires_grad=False)
        
        # # Periodic Reward Framework: theta for each leg
        # self.theta = torch.zeros((self.num_envs, 4), device=self.device)  # FL, FR, RL, RR
        # self._resample_behavior_params(torch.arange(self.num_envs, device=self.device))


    # def _resample_behavior_params(self, env_ids):
    #     if len(env_ids) == 0:
    #         return
        
    #     self.theta[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.theta_fl
    #     self.theta[env_ids, 1] = self.cfg.rewards.periodic_reward_framework.theta_fr
    #     self.theta[env_ids, 2] = self.cfg.rewards.periodic_reward_framework.theta_rl
    #     self.theta[env_ids, 3] = self.cfg.rewards.periodic_reward_framework.theta_rr

    # def _post_physics_step_callback(self):
    #     super()._post_physics_step_callback()
    #     env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
    #     # Periodic Reward Framework. resample phase and theta
    #     self._resample_behavior_params(env_ids)