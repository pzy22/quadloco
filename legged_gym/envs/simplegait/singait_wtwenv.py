from legged_gym.envs.simplegait.base_gait_env import BaseGaitEnv
from isaacgym.torch_utils import torch_rand_float, quat_apply, quat_from_angle_axis, quat_mul, quat_rotate_inverse, quat_conjugate
from legged_gym.utils.math import wrap_to_pi, get_scale_shift, quat_apply_yaw
import torch


class SinGait_WtwEnv(BaseGaitEnv):
    
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
        
        self.obs_buf = torch.cat((self.obs_buf,
                                  self.clock_inputs), dim=-1)

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
    #     # add perceptive inputs if not blind
    #     # if self.cfg.terrain.measure_heights:
    #     #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
    #     #     obs_buf = torch.cat((obs_buf, heights), dim=-1)

    #     # obs_buf = torch.cat((obs_buf,
    #     #                     self.last_actions), dim=-1)

    #     # obs_buf = torch.cat((obs_buf, 
    #     #                     self.clock_inputs), dim=-1)

    #     # privileged_obs_buf = obs_buf
    #     # if self.cfg.domain_rand.randomize_payload_mass:
    #     #     payload_scale, payload_shift = get_scale_shift(self.cfg.domain_rand.payload_mass_range)
    #     #     # print(privileged_obs_buf.shape)
    #     #     # print(self.payload.shape)
    #     #     # print(self.payload.shape)
    #     #     privileged_obs_buf = torch.cat((privileged_obs_buf,
    #     #                                    (self.payload - payload_shift) * payload_scale),
    #     #                                     dim=-1)

    #     # if self.cfg.domain_rand.randomize_com_displacement:
    #     #     com_scale, com_shift = get_scale_shift(self.cfg.domain_rand.com_displacement_range)
    #     #     privileged_obs_buf = torch.cat((privileged_obs_buf,
    #     #                                    (self.com_displacement - com_shift) * com_scale),
    #     #                                     dim=-1)

    #     # if self.cfg.domain_rand.randomize_friction:
    #     #     friction_scale, friction_shift = get_scale_shift(self.cfg.domain_rand.friction_range)
    #     #     privileged_obs_buf = torch.cat((privileged_obs_buf,
    #     #                                    (self.friction_coeffs - friction_shift) * friction_scale),
    #     #                                     dim=-1)
            
    #     # if self.cfg.domain_rand.randomize_motor_strength:
    #     #     motor_strength_scale, motor_strength_shift = get_scale_shift(self.cfg.domain_rand.motor_strength_range)
    #     #     privileged_obs_buf = torch.cat((privileged_obs_buf,
    #     #                                    (self.motor_strength_factors - motor_strength_shift) * motor_strength_scale),
    #     #                                     dim=-1)
            
    #     # if self.cfg.domain_rand.randomize_kp:
    #     #     kp_scale, kp_shift = get_scale_shift(self.cfg.domain_rand.kp_range)
    #     #     privileged_obs_buf = torch.cat((privileged_obs_buf,
    #     #                                    (self.Kp_factors - kp_shift) * kp_scale),
    #     #                                     dim=-1)
            
    #     # if self.cfg.domain_rand.randomize_kd:
    #     #     kd_scale, kd_shift = get_scale_shift(self.cfg.domain_rand.kd_range)
    #     #     privileged_obs_buf = torch.cat((privileged_obs_buf,
    #     #                                    (self.Kd_factors - kd_shift) * kd_scale),
    #     #                                     dim=-1)
            
    #     # if self.cfg.domain_rand.disturbance:
    #     #     disturbance_scale, disturbance_shift = get_scale_shift(self.cfg.domain_rand.disturbance_range)

    #     #     privileged_obs_buf = torch.cat((privileged_obs_buf,
    #     #                                    ((self.disturbance - disturbance_shift) * disturbance_scale).view(self.num_envs, -1)),
    #     #                                     dim=-1)

    #     # if self.cfg.domain_rand.push_robots:
    #     #     push_vel_scale, push_vel_shift = get_scale_shift(self.cfg.domain_rand.push_vel_xy_range)
    #     #     privileged_obs_buf = torch.cat((privileged_obs_buf,
    #     #                                    (self.rand_push_vels[:, :2] - push_vel_shift) * push_vel_scale),
    #     #                                     dim=-1)

    #     if self.add_noise:
    #         obs_buf = obs_buf.clone()
    #         obs_buf = 2 * torch.rand_like(obs_buf) - 1.0
    #         obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

    #     self.obs_buf = obs_buf
    #     #self.privileged_obs_buf = privileged_obs_buf


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

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

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
    

    # ------------ Walk These Ways 摘取 Reward Functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)


    def _reward_jump(self):
        # [NOTE] _reward_base_height 的替代版
        reference_heights = 0
        body_height = self.base_pos[:, 2] - reference_heights
        jump_height_target = self.cfg.rewards.base_height_target # + self.commands[:, 3]
        reward = - torch.square(body_height - jump_height_target)
        return reward

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self._get_base_heights()
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma))
        return reward / 4
    
    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.feet_vel, dim=2).view(self.num_envs, -1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma)))
        return reward / 4
    
    def _reward_dof_pos(self):
        # Penalize dof positions
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    

    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.joint_pos_target[:, :self.num_actions] - self.last_joint_pos_target[:, :self.num_actions])
        diff = diff * (self.last_actions[:, :self.num_dof] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.joint_pos_target[:, :self.num_actions] - 2 * self.last_joint_pos_target[:, :self.num_actions] + self.last_last_joint_pos_target[:, :self.num_actions])
        diff = diff * (self.last_actions[:, :self.num_dof] != 0)  # ignore first step
        diff = diff * (self.last_last_actions[:, :self.num_dof] != 0)  # ignore second step
        return torch.sum(diff, dim=1)
    

    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.feet_vel[:, :, 0:2], dim=2).view(self.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip

    def _reward_feet_contact_vel(self):
        reference_heights = 0
        near_ground = self.feet_pos[:, :, 2] - reference_heights < 0.03
        foot_velocities = torch.square(torch.norm(self.feet_vel[:, :, 0:3], dim=2).view(self.num_envs, -1))
        rew_contact_vel = torch.sum(near_ground * foot_velocities, dim=1)
        return rew_contact_vel
    
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :],
                                     dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_clearance_cmd_linear(self):
        # def _reward_feet_clearance_cmd_linear 去掉command控制
        # [NOTE] def _reward_foot_clearance 另一种实现
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.feet_pos[:, :, 2]).view(self.num_envs, -1)# - reference_heights
        target_swing_height = self.cfg.rewards.clearance_height_target * phases + 0.02  # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_swing_height - foot_height) * (1 - self.desired_contact_states)
        return torch.sum(rew_foot_clearance, dim=1)

    def _reward_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.feet_pos - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        desired_stance_width = 0.3
        desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.env.device).unsqueeze(0)

        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.env.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.cfg.gait.frequency
        x_vel_des = self.commands[:, 0:1]
        yaw_vel_des = self.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies) # frequencies.unsqueeze(1)
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies) # frequencies.unsqueeze(1)

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
    
    # ----------------- Walk These Ways 提取 Reward Functions 结束-----------------
    # 一些 额外 的 Reward Functions
    def _reward_joint_power(self):
        #Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

