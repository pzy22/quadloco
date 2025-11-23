from legged_gym.envs.simplegait.singait_env_cfg import SinGaitCfg, SinGaitCfgPPO

class A1SinGaitCfg(SinGaitCfg):
    class terrain( SinGaitCfg.terrain ):
        mesh_type = 'plane'  # plane, heightfield, trimesh

    class env( SinGaitCfg.env ):
        num_observations = 48

    class domain_rand( SinGaitCfg.domain_rand ):
        randomize_payload_mass = True
        randomize_com_displacement = True
        randomize_link_mass = False
        randomize_friction = True
        randomize_restitution = False
        randomize_motor_strength = True
        randomize_kp = True
        randomize_kd = True
        randomize_initial_joint_pos = True
        disturbance = True
        push_robots = True

    class init_state( SinGaitCfg.init_state ):
        pos = [0.0, 0.0, 0.34] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( SinGaitCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.0}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0

    class commands( SinGaitCfg.commands ):
            curriculum = True
            max_curriculum = 1.0
            num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
            resampling_time = 10. # time before command are changed[s]
            heading_command = True # if true: compute ang vel command from heading error
            class ranges( SinGaitCfg.commands.ranges):
                lin_vel_x = [-0.5, 0.5] # min max [m/s]
                lin_vel_y = [-0.5, 0.5]   # min max [m/s]
                ang_vel_yaw = [-1.57, 1.57]    # min max [rad/s]
                heading = [-3.14, 3.14]


    class asset( SinGaitCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/legged_gym/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up

    class rewards( SinGaitCfg.rewards ):
        class scales( SinGaitCfg.rewards.scales ):
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -1e-6 #-2.5e-7 #
            joint_power = -2e-5
            base_height = -50.0
            foot_clearance = -0.01 #-0.1 #
            action_rate = -0.01 #0.1#
            smoothness = -0.01 #-0.1#
            feet_air_time =  0.0
            collision = -0.0
            feet_stumble = -0.0
            stand_still = -0.5
            torques = -0.0
            dof_vel = -0.0#-1e-4
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0#-0.0
            torque_lidmits = -0.0

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.95 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        base_height_target = 0.26
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = -0.2
        kappa_gait_probs = 0.07

class A1SinGaitCfgPPO( SinGaitCfgPPO ):
    class algorithm( SinGaitCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( SinGaitCfgPPO.runner ):
        run_name = ''
        num_steps_per_env = 50
        max_iterations = 1000
        experiment_name = 'a1_singait_worew'