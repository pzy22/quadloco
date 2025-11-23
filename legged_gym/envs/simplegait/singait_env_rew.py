from legged_gym.envs.simplegait.base_gait_env import BaseGaitEnv
from isaacgym.torch_utils import torch_rand_float, quat_apply
from legged_gym.utils.math import wrap_to_pi, get_scale_shift
import torch


class SinGaitEnv(BaseGaitEnv):
    
    def _init_buffers(self):
        super()._init_buffers()
        
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        
        self.clock_inputs = torch.zeros(self.num_envs, 8, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, 
                                                requires_grad=False)


    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    ),dim=-1)
        
        self.privileged_obs_buf = self.obs_buf.clone()
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        # self.privileged_obs_buf = self.obs_buf.clone()
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec



    # def compute_observations(self):
        
    #     obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
    #                             self.base_ang_vel  * self.obs_scales.ang_vel,
    #                             self.projected_gravity,
    #                             self.commands[:, :3] * self.commands_scale,
    #                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
    #                             self.dof_vel * self.obs_scales.dof_vel,
    #                             self.actions
    #                             ),dim=-1)


    #     obs_buf = torch.cat((obs_buf,
    #                         self.last_actions), dim=-1)

    #     obs_buf = torch.cat((obs_buf, 
    #                         self.clock_inputs), dim=-1)

    #     privileged_obs_buf = obs_buf
    #     if self.cfg.domain_rand.randomize_payload_mass:
    #         payload_scale, payload_shift = get_scale_shift(self.cfg.domain_rand.payload_mass_range)
    #         # print(privileged_obs_buf.shape)
    #         # print(self.payload.shape)
    #         # print(self.payload.shape)
    #         privileged_obs_buf = torch.cat((privileged_obs_buf,
    #                                        (self.payload - payload_shift) * payload_scale),
    #                                         dim=-1)

    #     if self.cfg.domain_rand.randomize_com_displacement:
    #         com_scale, com_shift = get_scale_shift(self.cfg.domain_rand.com_displacement_range)
    #         privileged_obs_buf = torch.cat((privileged_obs_buf,
    #                                        (self.com_displacement - com_shift) * com_scale),
    #                                         dim=-1)

    #     if self.cfg.domain_rand.randomize_friction:
    #         friction_scale, friction_shift = get_scale_shift(self.cfg.domain_rand.friction_range)
    #         privileged_obs_buf = torch.cat((privileged_obs_buf,
    #                                        (self.friction_coeffs - friction_shift) * friction_scale),
    #                                         dim=-1)
            
    #     if self.cfg.domain_rand.randomize_motor_strength:
    #         motor_strength_scale, motor_strength_shift = get_scale_shift(self.cfg.domain_rand.motor_strength_range)
    #         privileged_obs_buf = torch.cat((privileged_obs_buf,
    #                                        (self.motor_strength_factors - motor_strength_shift) * motor_strength_scale),
    #                                         dim=-1)
            
    #     if self.cfg.domain_rand.randomize_kp:
    #         kp_scale, kp_shift = get_scale_shift(self.cfg.domain_rand.kp_range)
    #         privileged_obs_buf = torch.cat((privileged_obs_buf,
    #                                        (self.Kp_factors - kp_shift) * kp_scale),
    #                                         dim=-1)
            
    #     if self.cfg.domain_rand.randomize_kd:
    #         kd_scale, kd_shift = get_scale_shift(self.cfg.domain_rand.kd_range)
    #         privileged_obs_buf = torch.cat((privileged_obs_buf,
    #                                        (self.Kd_factors - kd_shift) * kd_scale),
    #                                         dim=-1)
            
    #     if self.cfg.domain_rand.disturbance:
    #         disturbance_scale, disturbance_shift = get_scale_shift(self.cfg.domain_rand.disturbance_range)

    #         privileged_obs_buf = torch.cat((privileged_obs_buf,
    #                                        ((self.disturbance - disturbance_shift) * disturbance_scale).view(self.num_envs, -1)),
    #                                         dim=-1)

    #     if self.cfg.domain_rand.push_robots:
    #         push_vel_scale, push_vel_shift = get_scale_shift(self.cfg.domain_rand.push_vel_xy_range)
    #         privileged_obs_buf = torch.cat((privileged_obs_buf,
    #                                        (self.rand_push_vels[:, :2] - push_vel_shift) * push_vel_scale),
    #                                         dim=-1)

    #     if self.add_noise:
    #         obs_buf = obs_buf.clone()
    #         obs_buf = 2 * torch.rand_like(obs_buf) - 1.0
    #         obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

    #     self.obs_buf = obs_buf
    #     self.privileged_obs_buf = privileged_obs_buf


    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -2., 2.)
        
        ##### 单一Gait 相关的参数更新 #####
        self._step_contact_targets()
        ##### -------------------- #####
        
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()



    def reset_idx(self,env_ids):
        super().reset_idx(env_ids)
        self.gait_indices[env_ids] = 0.0




    def _step_contact_targets(self):
        # 单一Gait Parameters
        frequency = self.cfg.gait.frequency
        durations = self.cfg.gait.duration
        target_gait = self.cfg.gait.target_gait  
        offsets = torch.tensor(self.cfg.gait.offsets[target_gait], device=self.device, requires_grad=False) # (num_envs,)
        
        self.gait_indices = torch.remainder(self.gait_indices + frequency * self.dt, 1.0)

        foot_indices = [self.gait_indices + offsets[0],
                        self.gait_indices + offsets[1],
                        self.gait_indices + offsets[2],
                        self.gait_indices + offsets[3]]

        #self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations
            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations)
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations) * (
                        0.5 / (1 - durations))

        for i in range(4):
            self.clock_inputs[:, i] = torch.sin(2 * torch.pi * foot_indices[i])
            self.clock_inputs[:, i + 4] = torch.cos(2 * torch.pi * foot_indices[i])

        # von mises distribution
        kappa = self.cfg.rewards.kappa_gait_probs
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

        smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        # noise_vec[48:60] = 0. # previous actions
        # noise_vec[60:68] = 0. # clock inputs
        return noise_vec



    # def _reward_action_smoothness_1(self):
    #     # Penalize changes in actions
    #     diff = torch.square(self.joint_pos_target[:, :self.num_actuated_dof] - self.last_joint_pos_target[:, :self.num_actuated_dof])
    #     diff = diff * (self.last_actions[:, :self.num_dof] != 0)  # ignore first step
    #     return torch.sum(diff, dim=1)
    
    # def _reward_action_smoothness_2(self):
    #     # Penalize changes in actions
    #     diff = torch.square(self.joint_pos_target[:, :self.num_actuated_dof] - 2 * self.last_joint_pos_target[:, :self.num_actuated_dof] + self.last_last_joint_pos_target[:, :self.num_actuated_dof])
    #     diff = diff * (self.last_actions[:, :self.num_dof] != 0)  # ignore first step
    #     diff = diff * (self.last_last_actions[:, :self.num_dof] != 0)  # ignore second step
    #     return torch.sum(diff, dim=1)