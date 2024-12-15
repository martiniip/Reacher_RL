import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import sys

def load_and_test_model(model_name, render_mode="rgb_array"):
    """
    Carga y prueba un modelo entrenado en un entorno.

    Args:
        model_name (str): Nombre del modelo a cargar (incluye el tipo de modelo y la ruta).
    """
    # Mapear el modelo al entorno correspondiente
    env_map = {
        "td3_pusher_model": "Pusher-v5",
        "td3_reacher_model": "Reacher-v5",
        "ppo_pusher_model": "Pusher-v5",
        "ppo_reacher_model": "Reacher-v5"
    }

    # Detectar el entorno basado en el modelo
    env_name = env_map.get(model_name, None)
    if not env_name:
        raise ValueError(f"No se reconoce el modelo especificado: {model_name}")

    num_eval_episodes = 4
    algorithm_name = model_name.split('_')[0].upper()
    environment_name = env_name.split('-')[0]
    video_folder = f"{algorithm_name}_{environment_name}_agent"

    # Crear el entorno
    env = gym.make(env_name, render_mode=render_mode)
    env = RecordVideo(env, video_folder=video_folder, name_prefix="eval", episode_trigger=lambda x: True)
    env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

    # Cargar el modelo
    print(f"Cargando el modelo: {model_name}...")

    if "ppo" in model_name.lower():
        from stable_baselines3 import PPO
        model = PPO.load(model_name)
    elif "td3" in model_name.lower():
        from stable_baselines3 import TD3
        model = TD3.load(model_name)
    else:
        raise ValueError(f"No hay soporte para el modelo especificado: {model_name}")

    # Iniciar la simulación
    for episode_num in range(num_eval_episodes):
        obs, info = env.reset()

        episode_over = False
        while not episode_over:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated

    env.close()
    print("Prueba finalizada con éxito.")
    print(f'Episode time taken: {env.time_queue}')
    print(f'Episode total rewards: {env.return_queue}')
    print(f'Episode lengths: {env.length_queue}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba un modelo en un entorno específico.")
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo o nombre del modelo.")

    args = parser.parse_args()

    # Probar el modelo
    load_and_test_model(args.model)
