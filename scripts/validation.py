import numpy as np

import tensorflow as tf
#from tensorflow.keras.callbacks import EarlyStopping
EarlyStopping = tf.keras.callbacks.EarlyStopping
import sys, os
#sys.path.append('./neutral_networks.py')  # lub ścieżka do neutral_networks.py
scripts_dir = os.path.abspath(os.path.join('neutral_networks.py', os.pardir))
sys.path.insert(0, scripts_dir)


import neutral_networks as nn

# 1) Wczytaj i połącz zbiór (np. train+val) lub użyj dataset.npy:
train = np.load('./data/datasets/standard_split/color/128x128/train_set.npy', allow_pickle=True)
val   = np.load('./data/datasets/standard_split/color/128x128/val_set.npy', allow_pickle=True)
data_all = np.concatenate([train, val], axis=0)

# 2) Rozpakuj X, y
X, y = nn.unpacking_data(data_all)

# 3) Przygotuj model CNN i callback EarlyStopping
n_classes = y.shape[1]
input_shape = X.shape[1:]  # (H, W, C)
model_cnn = nn.create_model_cnn(
    n_classes=n_classes,
    input_shape=input_shape,
    optimizer='adam',
    func_activation='relu',
    kernel_initializer='he_uniform'
)
callbacks = [EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]

#4) 5-krotna walidacja
scores_cnn = nn.cross_validation(
    n_splits=10,
    X=X, y=y,
    model=model_cnn,
    epochs=100,
    batch_size=16,
    callbacks=callbacks
)
print("CNN fold scores:", scores_cnn, "mean:", np.mean(scores_cnn))

# 5) Analogicznie MLP (rozpłaszczenie X przed CV)
X_flat = nn.flatten_data(X, img_type='color')  # jeśli to grayscale; zmień w razie potrzeby
model_mlp = nn.create_model_mlp(
    n_classes=n_classes,
    input_shape=(X_flat.shape[1],),
    optimizer='adam',
    func_activation='relu',
    kernel_initializer='he_uniform'
)
scores_mlp = nn.cross_validation(
    n_splits=10,
    X=X_flat, y=np.argmax(y, axis=1),
    model=model_mlp,
    epochs=100,
    batch_size=16,
    callbacks=callbacks
)
print("MLP  fold scores:", scores_mlp, "mean:", np.mean(scores_mlp))
