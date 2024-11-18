import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

def train_ppo(env_name, total_timesteps, model_path):
    # Crear el entorno
    env = gym.make(env_name)

    # Crear el modelo PPO
    model = PPO(MlpPolicy, env, verbose=1)

    # Entrenar el modelo
    model.learn(total_timesteps=total_timesteps)

    # Guardar el modelo entrenado
    model.save(model_path)

    env.close()

if __name__ == "__main__":
    train_ppo("Reacher-v5", total_timesteps=1000, model_path="ppo_reacher_model")