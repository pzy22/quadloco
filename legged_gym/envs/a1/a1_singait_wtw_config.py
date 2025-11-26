from legged_gym.envs.simplegait.singait_wtwenv_cfg import SinGaitWtwCfg, SinGaitWtwCfgPPO

PROPRIOCEPTION_DIM = 48
LAST_ACTION_DIM = 12
CLOCK_INPUTS_DIM = 8
HIEGHET_MEASURE_DIM = 235-48
CONTACT_DIM = 4
BODY_HEIGHT_DIM = 1
class A1SinGaitWtwCfg(SinGaitWtwCfg):
    class gait( SinGaitWtwCfg.gait ):
        frequency = 2.0  # [Hz]
        duration = 0.5 
        target_gait  = "bound"

    
    class env( SinGaitWtwCfg.env ):
        num_observations = PROPRIOCEPTION_DIM + LAST_ACTION_DIM + CLOCK_INPUTS_DIM
        num_privileged_obs = PROPRIOCEPTION_DIM + LAST_ACTION_DIM + CLOCK_INPUTS_DIM + CONTACT_DIM + BODY_HEIGHT_DIM + HIEGHET_MEASURE_DIM

    class terrain( SinGaitWtwCfg.terrain ):
        mesh_type = 'plane'  # plane, heightfield, trimesh
        measure_heights = True

    class domain_rand( SinGaitWtwCfg.domain_rand ):
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

    class init_state( SinGaitWtwCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
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

    class control( SinGaitWtwCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.0}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0

    class commands( SinGaitWtwCfg.commands ):
            curriculum = True
            max_curriculum = 1.0
            num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
            resampling_time = 10. # time before command are changed[s]
            heading_command = True # if true: compute ang vel command from heading error
            class ranges( SinGaitWtwCfg.commands.ranges):
                lin_vel_x = [-0.5, 0.5] # min max [m/s]
                lin_vel_y = [-0.5, 0.5]   # min max [m/s]
                ang_vel_yaw = [-1.57, 1.57]    # min max [rad/s]
                heading = [-3.14, 3.14]


    class asset( SinGaitWtwCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/legged_gym/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up

    class rewards( SinGaitWtwCfg.rewards ):
        class scales( SinGaitWtwCfg.rewards.scales ):
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0#-0.02
            ang_vel_xy = -0.05#-0.001
            orientation = -0.2
            dof_acc = -1e-6 #-2.5e-7 #
            dof_vel = -0.0#-1e-4
            base_height = - 0.0
            jump = 50.0 # 和base_height一样的implementation

            feet_contact_forces = 0.0
            feet_slip = -0.04
            action_smoothness_1 = -0.1
            action_smoothness_2 = -0.1            
            raibert_heuristic = -10.0
            foot_clearance = -0.0
            footswing_cmd_linear = -30.0 # target 是 foot swing height
            feet_air_time = 0.0
            tracking_contacts_shaped_force = 4.0
            tracking_contacts_shaped_vel = 4.0
            collision = -5.0

            action_rate = -0.01
            stand_still = -0.0

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_ji22_style = True
        sigma_rew_neg = 0.02

        kappa_gait_probs = 0.07
        gait_force_sigma = 100.
        gait_vel_sigma = 10

        clearance_height_target = -0.0
        footswing_height_target = 0.1

        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.95 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        base_height_target = 0.26
        max_contact_force = 100. # forces above this value are penalized
        



class A1SinGaitWtwCfgPPO(SinGaitWtwCfgPPO):
    class algorithm( SinGaitWtwCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( SinGaitWtwCfgPPO.runner ):
        run_name = ''
        num_steps_per_env = 50
        max_iterations = 1500
        experiment_name = 'a1_singait_wtw'