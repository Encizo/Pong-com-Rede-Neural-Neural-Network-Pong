import numpy as np
from tensorflow.keras import models, layers
import os

RESULTS_FILE = "resultados.txt"

class Rede:
    @staticmethod
    def train_neural_network():
        model = models.Sequential([
            layers.Dense(8, input_dim=4, activation='relu'),  # Entradas: [posição raquete, distância vertical, distância horizontal, direção bola]
            layers.Dense(8, activation='relu'),
            layers.Dense(2, activation='sigmoid')  # Saídas: [mover cima, mover baixo]
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def generate_individuals(parent_model):
        new_individuals = []
        for i in range(5):
            new_model = models.clone_model(parent_model)
            new_model.set_weights(parent_model.get_weights())
            for layer in new_model.layers:
                if isinstance(layer, layers.Dense):
                    weights, biases = layer.get_weights()
                    weights += np.random.normal(scale=0.15, size=weights.shape)
                    layer.set_weights([weights, biases])
            new_individuals.append(new_model)
        return new_individuals

    @staticmethod
    def clear_results():
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)
