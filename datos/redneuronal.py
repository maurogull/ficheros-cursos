import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input

# Datos simulados: [edad, visitas, tiempo_en_pagina]
X = np.array([
    [25, 3, 45], [40, 10, 80], [22, 1, 15],
    [35, 5, 60], [50, 12, 100], [30, 2, 20],
    [26, 3, 46], [41, 9, 83], [23, 2, 14],
    [35, 4, 61], [50, 12, 104], [31, 2, 19]
])
y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])  # 1 = compra, 0 = no compra

modelo = Sequential([
    Input(shape=(3,)),  
    layers.Dense(8, activation='relu'),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modelo.fit(X, y, epochs=30, verbose=0)

print(modelo.predict(np.array([[30, 20, 55]])))