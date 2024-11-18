import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

# Crear el entorno Reacher-v5
env = gym.make('Reacher-v5', render_mode='human')

# Crear el modelo PPO con la política de red neuronal MLP
##model = PPO(MlpPolicy, env, verbose=1)

# Entrenar el modelo
##model.learn(total_timesteps=10000000)  # Número de pasos de entrenamiento

# Guardar el modelo entrenado
##model.save("ppo_reacher_model")  # Guarda el modelo en el archivo "ppo_reacher_model"

# Cargar el modelo entrenado desde el archivo
model = PPO.load("ppo_reacher_model")

# Reiniciar el entorno
obs, info = env.reset()

done = False
while not done:
    # Renderizar el entorno en pantalla
    env.render()

    # Usar el modelo cargado para predecir la siguiente acción
    action, _states = model.predict(obs)

    # Realizar un paso en el entorno usando la acción predicha por el modelo
    obs, reward, done, truncated, info = env.step(action)

# Cerrar el entorno al finalizar
env.close()