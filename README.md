«Reacher» es un brazo robótico de dos articulaciones. El objetivo es mover el efector final del robot (llamado punta del dedo) cerca de un objetivo que se genera en una posición aleatoria.

# entrenar un modelo
-python TD3.py pusher | reacher

-python PPO.py pusher | reacher


# Cargar modelo entrenado
-python test.py --model ppo_reacher_model

-python test.py --model td3_reacher_model
- python test.py --model ppo_reacher_model
- python test.py --model td3_reacher_model
