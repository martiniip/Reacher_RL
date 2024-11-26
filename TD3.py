import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy

def train_td3(env_name, total_timesteps, model_path, log_dir):
    # Crear el entorno
    env = gym.make(env_name)

    # Crear el modelo TD3 con TensorBoard logging
    model = TD3(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)

    # Entrenar el modelo
    model.learn(total_timesteps=total_timesteps, tb_log_name="TD3_run")

    # Guardar el modelo entrenado
    model.save(model_path)

    env.close()

if __name__ == "__main__":
    train_td3("Reacher-v5", total_timesteps=100000, model_path="td3_reacher_model", log_dir="./td3_tensorboard/")
