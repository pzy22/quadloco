静止的时候有些抖动
```python
    class rewards( BaseGaitCfg.rewards ):
        class scales( BaseGaitCfg.rewards.scales ):
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -10.0
            foot_clearance = -0.0#-0.1 #-0.01
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time =  0.0
            collision = -0.0
            feet_stumble = -0.0
            stand_still = -0.01
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
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = -0.2

```


静止的时候有些抖动
```python
    class rewards( BaseGaitCfg.rewards ):
        class scales( BaseGaitCfg.rewards.scales ):
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -50.0
            foot_clearance = -0.01#-0.1 #-0.01
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time =  0.0
            collision = -0.0
            feet_stumble = -0.0
            stand_still = -0.05
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
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = -0.2

```
