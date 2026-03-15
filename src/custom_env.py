from logging import config

from metadrive.envs.metadrive_env import MetaDriveEnv
import numpy as np

# Custom reward function for the straight line task
# positive: move forward, speed limit,
# negative: collision, speeding, off road, erratic driving like switching lanes unnrecessarily
class CustomMetaDriveEnv(MetaDriveEnv):
    # Setup environment with custom configuration
    def __init__(self, config=None):
        self.target_speed = 30.0
        if config and "target_speed" in config:
            self.target_speed_limit = config.pop("target_speed")

        default_config = dict(
            use_render=False,
            manual_control=False,
            traffic_density=0.0,
            random_agent_model=False,
            top_down_camera_initial_x=0,
            top_down_camera_initial_y=0,
            image_observation=False,
            sensors=dict(),
            agent_observation=None
            )
        if config:
            default_config.update(config)
        super(CustomMetaDriveEnv, self).__init__(default_config)

    def reward_function(self, vehicle_id):
        """
        Custom reward function.
        """

        vehicle =  self.agents[vehicle_id]
        step_info = dict()

        # Get current stats
        current_speed = vehicle.speed * 2.237  # m/s to mph
        target_speed = 30.0  # (mph)

        # Speed reward: Positive for keeping target speed, negative for speeding
        if current_speed <= target_speed:
            speed_reward = current_speed / target_speed
        else:
            speed_reward = 1.0 - (current_speed - target_speed) / 10.0

        total_reward = max(-1.0, speed_reward)  # Ensure reward is not less than -1

        if current_speed < 0.1:
            total_reward -= 0.1

        # Lane keeping penalty increases the further the vechicle is from the center of the lane
        _, lateral_dist = vehicle.navigation.current_lane.local_coordinates(vehicle.position)
        lane_penalty = (abs(lateral_dist) / vehicle.navigation.get_current_lane_width()) * 1.0
        total_reward -= lane_penalty

        # Penalise for erratic driving
        steering_action = abs(vehicle.last_current_action[0][0]) if vehicle.last_current_action else 0.0
        steering_penalty = 0.05 * steering_action 
        total_reward -= steering_penalty

        # Penalise heavily for collisions and off road
        if vehicle.crash_vehicle or vehicle.crash_object or vehicle.crash_human or vehicle.crash_sidewalk:
            total_reward = -10.0
            step_info["is_crash"] = True
        elif self._is_out_of_road(vehicle):
            total_reward = -10.0
            step_info["is_out_of_road"] = True

        if self._is_arrive_destination(vehicle):
            total_reward = 10.0

        step_info["step_reward"] = total_reward
        step_info["speed_reward"] = speed_reward
        step_info["lane_penalty"] = lane_penalty
        step_info["steering_penalty"] = steering_penalty
        return total_reward, step_info
