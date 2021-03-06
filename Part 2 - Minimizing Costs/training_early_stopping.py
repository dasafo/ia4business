# Inteligencia Artificial aplicada a Negocios y Empresas - Caso Práctico 2
# Entrenamiento de la IA

# Instalación de Keras
# conda install -c conda-forge keras

# Importar las librerías y el resto de ficheros de python
import os
import numpy as np
import random as rn
import environment
import brain
import dqn

# Establecer semillas para la reproducibilidad del experimento
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# CONFIGURACIÓN DE LOS PARÁMETROS
epsilon = .3
number_actions = 5
direction_boundary = (number_actions - 1) / 2
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5

# CONSTRUCCIÓN DEL ENTORNO CREANDO UN OBJETO DE LA CLASE ENVIRONMENT CLASS
env = environment.Environment(optimal_temperature = (18.0, 24.0),initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

# CONSTRUCCIÓN DEL CEREBRO CREADO UN OBJETO DE LA CLASE BRAIN
brain = brain.Brain(learning_rate = 0.00001, number_actions = number_actions)

# CONSTRUCCIÓN DEL MODELO DE DQN CREANDO UN OBJETO DE LA CLASE DQN 
dqn = dqn.DQN(max_memory = max_memory, discount_factor = 0.9)

# ELECCIÓN DEL MODO DE ENTRENAMIENTO
train = True

# Entrenamiento de la IA
env.train = train
model = brain.model
early_stopping = True
patience = 10
best_total_reward = -np.inf
patience_count = 0
if (env.train):
    # STARTING EPOCH BUCLE (1 Epoch = 5 Mouths)
    for epoch in range(1, number_epochs):
        # STARTING ENVIRONMEN VARIABLES AND TRAINING BUCLE
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)
        game_over = False
        current_state, _, _ = env.observe() #we only want current_state return from give_env method
        timestep = 0
        # INICIALIZATION TIMESTEPS BUCLE(Timestep = 1 minute) AT ONE EPOCH
        while ((not game_over) and timestep <= 5 * 30 * 24 * 60):
            # RUNNING NEXT ACTION BY EXPLORATION
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step
            
            # RUNNING NEXT ACTION BY EXPLOTATION
            else:
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step
               
            # UPDATING ENVIRONMENT AND REACHING NEXT STATE
            next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))
            total_reward += reward

            # SAVING NEW TRANSITION IN MEMORY
            dqn.remember([current_state, action, reward, next_state], game_over)

            # GETTING INPUTS AND TARGETS BLOCKS
            inputs, targets = dqn.get_batch(model, batch_size)

            # CALCULATING LOOST FUNCTION WITH THE WHOLE INPUT AND TARGET BLOCK
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state
        # IMPRIMIR EL RESULTADO DE ENTREAMIENTO PARA CADA EPOCH
        print("\n")
        print("Epoch: {:03d}/{:03d}".format(epoch, number_epochs))
        print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
        print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))
        # EARLY STOPPING
        if (early_stopping):
            if (total_reward <= best_total_reward):
                patience_count += 1
            elif (total_reward > best_total_reward):
                best_total_reward = total_reward
                patience_count = 0
            if (patience_count >= patience):
                print("Early Stopping")
                break
        # GUARDAR EL MODELO
        model.save("model.h5")
