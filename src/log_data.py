import pygame
import numpy as np
import pandas as pd
from metadrive import MetaDriveEnv
import os

csv_filename = input("Enter CSV filename: ").strip()
if not csv_filename.endswith(".csv"):
    csv_filename += ".csv"
print(f"Data will be saved to: {csv_filename}")

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("Connect joystick.")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
print("Joystick: {}".format(joystick.get_name()))

env_config = {
    "map": "S", 
    "use_render": True,
    "manual_control": False,
    "random_lane_index": True,
}
env = MetaDriveEnv(env_config)
obs, info = env.reset()
observations = []
actions = []

clock = pygame.time.Clock()
running = True

print("LOGGING DATA")
while running:
    pygame.event.pump()

    # Action mapping
    steer = -joystick.get_axis(0) ** 3
    l2_brake = (joystick.get_axis(4) + 1.0) / 2.0
    r2_gas = (joystick.get_axis(5) + 1.0) / 2.0
    
    throttle = r2_gas - l2_brake 

    action = np.array([steer, throttle])

    # Env step
    next_obs, reward, terminated, truncated, info = env.step(action)

    # save data
    observations.append(obs)
    actions.append(action)

    obs = next_obs

    if terminated or truncated:
        obs, info = env.reset()

    # Exit with square / x
    if joystick.get_button(2):
        print("\nStopping simulation and saving data...")
        running = False

    # Lock loop to 10 iterations per second (10 FPS)
    clock.tick(10)

env.close()
pygame.quit()

obs_df = pd.DataFrame(observations)
actions_df = pd.DataFrame(actions, columns=["steer", "throttle"])
dataset = pd.concat([obs_df, actions_df], axis=1)
os.makedirs("../data", exist_ok=True)
dataset.to_csv(f"../data/{csv_filename}", index=False)
print(f"Data saved to ../data/{csv_filename}")