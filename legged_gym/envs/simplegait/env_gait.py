from legged_gym.envs.base.legged_robot import LeggedRobot



class SingleGaitEnv(LeggedRobot):
    def __init__(self, cfg, sim_device, headless):
        super().__init__(cfg, sim_device, headless)