# Inteligencia Artificial aplicada a Negocios y Empresas - Caso Práctico 2
# Construcción del cerebro

# Importar las librerías
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

# CONSTRUCCIÓN DEL CEREBRO

class Brain(object):
    
    # CONSTRUCCIÓN DE UNA RED NEURONAL TOTALMENTE CONECTADA EN EL MÉTODO DE INICIALIZACIÓN
    
    def __init__(self, learning_rate = 0.001, number_actions = 5):
        self.learning_rate = learning_rate
        
        # CONSTRUCCIÓN DE LA CAPA DE ENTRADA COMPUESTA DE LOS ESTADOS DE ETRADA
        states = Input(shape = (3,))
        
        # CONSTRUCCIÓN DE PRIMERA CAPA OCULTAS TOTALMENTE CONECTADA CON DROPOUT ACTIVADO
        x = Dense(units = 64, activation = 'sigmoid')(states)
        x = Dropout(rate = 0.1)(x)

        # CONSTRUCCIÓN DE SEGUNDA CAPA OCULTAS TOTALMENTE CONECTADA CON DROPOUT ACTIVADO
        y = Dense(units = 32, activation = 'sigmoid')(x)
        y = Dropout(rate = 0.1)(y)

        # CONSTRUCCIÓN DE LA CAPA DE SALIDA, TOTALMENTE CONECTADA A LA ÚLTIMA CAPA OCULTA
        q_values = Dense(units = number_actions, activation = 'softmax')(y)
        
        # ENSAMBLAR LA ARQUITECTURA COMPLETA EN UN MODELO DE KERAS
        self.model = Model(inputs = states, outputs = q_values)
        
        # COMPILAR EL MODELO CON LA FUNCIÓN DE PÉRDIDAS DE ERROR CUADRÁTICO MEDIO Y EL OPTIMIZADOR ELEGIDO
        self.model.compile(loss = 'mse', optimizer = Adam(lr = learning_rate))
