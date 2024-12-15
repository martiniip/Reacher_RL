import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.monitor import Monitor
import sys
import os

def train_ppo(env_name, total_timesteps, model_path, log_dir, log_name):
    # Crear el entorno
    env = Monitor(gym.make(env_name))

    # Crear el modelo PPO
    model = PPO(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)

    # Entrenar el modelo
    model.learn(total_timesteps=total_timesteps, tb_log_name=log_name)

    # Guardar el modelo entrenado
    model.save(model_path)

    env.close()

if __name__ == "__main__":
    # Verificar la entrada por terminal
    if len(sys.argv) != 2:
        print("Uso: python script.py <pusher|reacher>")
        sys.exit(1)

    # Mapear el nombre simplificado al nombre del entorno completo
    env_short_name = sys.argv[1].lower()
    env_map = {
        "pusher": "Pusher-v5",
        "reacher": "Reacher-v5"
    }

    if env_short_name not in env_map:
        print("Error: El entorno debe ser 'pusher' o 'reacher'.")
        sys.exit(1)

    # Obtener el nombre del entorno completo
    env_name = env_map[env_short_name]

    # Definir parámetros de entrenamiento
    total_timesteps = 5000000
    model_path = f"ppo_{env_short_name}_model"

    # Crear el directorio de registro según el modelo y el entorno
    log_dir = f"./ppo_logs/{env_short_name}/"
    os.makedirs(log_dir, exist_ok=True)

    # Crear el nombre del registro para TensorBoard
    log_name = f"PPO_{env_short_name}_run"

    # Entrenar el modelo
    train_ppo(env_name, total_timesteps, model_path, log_dir, log_name)
