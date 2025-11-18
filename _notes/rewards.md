# Reward 函数说明（中文）

下面是 collect_env.py 中各个 reward function 的简要说明与备注。

| 函数名 | 功能（中文） | 计算要点 / 备注 |
|---|---:|---|
| _reward_lin_vel_z | 惩罚 z 方向线速度 | 返回基座在 z 轴速度的平方（减少上下颠簸） |
| _reward_ang_vel_xy | 惩罚 roll/pitch 角速度 | 返回基座在 x,y 轴角速度平方和（抑制翻滚、俯仰） |
| _reward_orientation | 惩罚非平坦姿态 | 返回投影重力在 x,y 分量的平方和（鼓励基座水平） |
| _reward_base_height | 惩罚基座高度偏离目标 | 计算基座 z 与测量高度差的均值，与 cfg 中目标高度的平方误差 |
| _reward_torques | 惩罚使用大力矩 | 返回所有关节力矩平方和（鼓励低力矩） |
| _reward_dof_vel | 惩罚关节速度 | 返回关节速度平方和 |
| _reward_dof_acc | 惩罚关节加速度 | 用 (last_dof_vel - dof_vel)/dt 的平方和作为加速度惩罚 |
| _reward_action_rate | 惩罚动作变化速率 | 返回 last_actions 与 actions 差值的平方和（平滑动作） |
| _reward_collision | 惩罚指定刚体的碰撞 | 统计 penalised_contact_indices 上法向力大于阈值的碰撞次数 |
| _reward_termination | 终止奖励/惩罚 | 在 reset 且不是 time-out 的情况下触发（终止相关的奖励/惩罚） |
| _reward_dof_pos_limits | 惩罚接近/超出关节位置限制 | 计算超出下/上限的距离并求和（越接近/越超出惩罚越大） |
| _reward_dof_vel_limits | 惩罚接近速度限制 | 绝对速度减软限制后截断到 [0,1] 再求和（限制最大误差为 1 rad/s） |
| _reward_torque_limits | 惩罚接近力矩限制 | 绝对力矩减软力矩限制后截断并求和 |
| _reward_tracking_lin_vel | 线速度跟踪奖励（xy） | 以 exp(-lin_vel_error / sigma) 给出，误差越小奖励越高 |
| _reward_tracking_ang_vel | 角速度跟踪奖励（yaw） | 以 exp(-ang_vel_error / sigma) 给出，对 yaw 角速度误差建模 |
| _reward_feet_air_time | 奖励步态空中时间（长步） | 过滤不可靠 contact，记录脚离地时间，在“第一次着地”时按离地时间奖励；仅在有速度命令时给奖励 |
| _reward_stumble | 惩罚脚部撞击竖直面 | 当水平接触力大于 5 倍竖直分量时计为绊倒（布尔 any） |
| _reward_stand_still | 在零命令时惩罚移动 | 当命令近零时，惩罚关节偏离默认位置的绝对和（鼓励静止） |
| _reward_feet_contact_forces | 惩罚过高脚部接触力 | 对脚部接触力超过 cfg.rewards.max_contact_force 的部分求和惩罚 |