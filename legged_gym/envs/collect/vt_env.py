
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.collect.cmd_curriculum_v1 import RewardThresholdCurriculum

import numpy as np
import torch

class VelocityTrackingEnv_v1(LeggedRobot):
    
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._init_command_distribution(np.arange(self.num_envs))

    def _init_command_distribution(self, env_ids):
        if self.cfg.commands.curriculum:
            self.curriculum = RewardThresholdCurriculum(seed=self.cfg.commands.curriculum_seed,
                                                        x_vel=(self.cfg.commands.limit_vel_x[0],
                                                            self.cfg.commands.limit_vel_x[1], 51),
                                                        y_vel=(self.cfg.commands.limit_vel_y[0],
                                                            self.cfg.commands.limit_vel_y[1], 2),
                                                        yaw_vel=(self.cfg.commands.limit_vel_yaw[0],
                                                                self.cfg.commands.limit_vel_yaw[1], 51))
            
            self.env_command_bins = np.zeros(len(env_ids), dtype=np.int)
            
            low = np.array(
                [self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_y[0],
                self.cfg.commands.ang_vel_yaw[0]])
            high = np.array(
                [self.cfg.commands.lin_vel_x[1], self.cfg.commands.lin_vel_y[1],
                self.cfg.commands.ang_vel_yaw[1]])
            
            self.curriculum.set_to(low=low, high=high)

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        if self.cfg.commands.curriculum:
            timesteps = int(self.cfg.commands.resampling_time / self.dt)
            ep_len = timesteps
            lin_vel_rewards = self.command_sums["tracking_lin_vel"][env_ids] / ep_len
            ang_vel_rewards = self.command_sums["tracking_ang_vel"][env_ids] / ep_len
            lin_vel_threshold = self.cfg.commands.forward_curriculum_threshold * self.reward_scales["tracking_lin_vel"]
            ang_vel_threshold = self.cfg.commands.yaw_curriculum_threshold * self.reward_scales["tracking_ang_vel"]

            old_bins = self.env_command_bins[env_ids.cpu().numpy()]
            # update step just uses train env performance (for now)
            self.curriculum.update(old_bins[env_ids.cpu().numpy()],
                                lin_vel_rewards[env_ids].cpu().numpy(),
                                ang_vel_rewards[env_ids].cpu().numpy(), lin_vel_threshold,
                                ang_vel_threshold, local_range=0.5, )
            
            new_commands, new_bin_inds = self.curriculum.sample(batch_size=len(env_ids))

            self.env_command_bins[env_ids.cpu().numpy()] = new_bin_inds
            self.commands[env_ids, :3] = torch.Tensor(new_commands).to(self.device)
            self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
            # reset command sums
            for key in self.command_sums.keys():
                self.command_sums[key][env_ids] = 0.
        else:
            super()._resample_commands(env_ids)


    def update_command_curriculum(self, env_ids):
        return 