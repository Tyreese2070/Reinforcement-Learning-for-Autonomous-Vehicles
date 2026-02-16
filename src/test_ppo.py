from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3 import PPO

MODEL_LOCATION = "../models/ppo_baseline"

def test_agent():
    env = MetaDriveEnv(dict(
        use_render=True,
        traffic_density=0.0,
        map="S",
        manual_control=False,
    ))

    try:
        model = PPO.load(MODEL_LOCATION)
        print("Loaded model")
    except FileNotFoundError:
        print("Train the model.")

    obs, info = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    test_agent()