import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Učitavanje podataka
x = np.load('observations.npy', allow_pickle=True)
y = np.load('actions.npy', allow_pickle=True)

num_classes = len(np.unique(y))
y = tf.keras.utils.to_categorical(y, num_classes)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100,batch_size=8)

val_loss, val_accuracy = model.evaluate(x_val, y_val)
print(f'Validation accuracy: {val_accuracy:.3f}')

model.save('trained_model.h5')
