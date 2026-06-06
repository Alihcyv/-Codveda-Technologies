%%writefile mnist_digit_pipline.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

def plot_history(history, title):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test (Val) Loss')
    plt.title(f'{title} - Loss Evolution')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Test (Val) Acc')
    plt.title(f'{title} - Accuracy Evolution')
    plt.legend()
    plt.show()

print('\n' + '#' * 30 + ' Deep Neural Network ' + '#' * 30 + '\n')
model = models.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
        
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
        
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
history = model.fit(X_train, y_train, epochs=8, batch_size=64, 
                        validation_data=(X_test, y_test), verbose=1)
    
plot_history(history, "DNN")


print('\n' + '#' * 30 + ' Convolutional Neural Network ' + '#' * 30 + '\n')
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
        
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
        
    layers.Flatten(),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
history = model.fit(X_train_cnn, y_train, epochs=8, batch_size=64, 
                        validation_data=(X_test_cnn, y_test), verbose=1)
    
plot_history(history, "CNN")
