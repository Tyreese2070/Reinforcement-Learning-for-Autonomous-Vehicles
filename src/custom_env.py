from metadrive.envs.metadrive_env import MetaDriveEnv
import numpy as np

# Custom reward function for the straight line task
# positive: move forward, speed limit,
# negative: collision, speeding, off road, erratic driving like switching lanes
class StraightLineEnv(MetaDriveEnv):
    pass

