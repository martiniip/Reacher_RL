import argparse
import gymnasium as gym

def load_and_test_model(model_name, render_mode="human"):
    """
    Carga y prueba un modelo entrenado en el entorno Reacher-v5.

    Args:
        model_name (str): Nombre del modelo a cargar (incluye el tipo de modelo y la ruta).
    """
    # Crear el entorno
    env_name = "Reacher-v5"
    env = gym.make(env_name, render_mode=render_mode)

    # Determinar cómo cargar el modelo según su tipo o nombre
    print(f"Cargando el modelo: {model_name}...")

    # Ejemplo: Detectar tipo de modelo 
    if "ppo" in model_name.lower():
        from stable_baselines3 import PPO
        model = PPO.load(model_name)
    elif "td3" in model_name.lower():
        from stable_baselines3 import TD3
        model = TD3.load(model_name)
    else:
        # Lógica personalizada para otros modelos
        raise ValueError(f"No hay soporte para el modelo especificado: {model_name}")

    # Iniciar la simulación
    obs, info = env.reset()
    done = False

    while not done:
        env.render()

        # Predecir la acción con el modelo cargado
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

    env.close()
    print("Prueba finalizada con éxito.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba un modelo entrenado en el entorno Reacher-v5.")
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo a cargar.")

    args = parser.parse_args()

    # Llamar a la función para cargar y probar el modelo
    load_and_test_model(args.model)
