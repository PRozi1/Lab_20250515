import os
import numpy as np
import tensorflow as tf

from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics 

Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
np_utils = tf.keras.utils
EarlyStopping = tf.keras.callbacks.EarlyStopping


def load_array_from_file(file_name):
    # ensure .npy extension
    if not file_name.endswith('.npy'):
        file_name += '.npy'
    return np.load(file_name, allow_pickle=True)


def load_dataset(main_path_sets):
    train_set = load_array_from_file(f'{main_path_sets}/train_set')
    val_set   = load_array_from_file(f'{main_path_sets}/val_set')
    test_set  = load_array_from_file(f'{main_path_sets}/test_set')
    return train_set, val_set, test_set


def unpacking_data(data):
    X = data[:, 0]
    y = data[:, 1]
    X = np.array(list(X), np.float32)
    y = np_utils.to_categorical(y)
    return X, y


def count_samples_and_classes(datasets):
    """
    datasets: dict of name->(X,y) arrays (before one-hot)
    Prints sample count and unique classes for each dataset.
    """
    print("\nZestaw danych - liczba próbek i klas:")
    for name, data in datasets.items():
        X_raw, y_raw = data
        n_samples = X_raw.shape[0]
        n_classes = len(np.unique(y_raw))
        print(f"{name}: samples={n_samples}, classes={n_classes}")


def flatten_data(dataset, img_type):
    # dataset: np.ndarray of shape (n_samples, H, W[, C])
    shape = dataset.shape
    if img_type == 'color':
        n, h, w, c = shape
        size = h * w * c
    else:
        n, h, w = shape
        size = h * w
    return dataset.reshape((n, size))


def create_model_mlp(n_classes, input_shape,
                     optimizer='rmsprop', func_activation='relu',
                     kernel_initializer='random_uniform'):
    model = Sequential([
        Dense(128, input_shape=input_shape,
              activation=func_activation,
               kernel_initializer=kernel_initializer),
        Dense(n_classes, activation='softmax',
              kernel_initializer='random_uniform'),
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def create_model_cnn(n_classes, input_shape,
                     optimizer='adam', func_activation='relu',
                     kernel_initializer='random_uniform'):
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation=func_activation,
               kernel_initializer=kernel_initializer, input_shape=input_shape),
        Conv2D(32, (3, 3), padding='same', activation=func_activation,
               kernel_initializer=kernel_initializer),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation=func_activation,
              kernel_initializer=kernel_initializer),
        Dense(n_classes, activation='softmax'),
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def main():
    # ustawienia użytkownika
    data_split_type = 'standard_split'
    neural_network  = 'mlp'  # 'mlp' lub 'cnn'
    img_type        = 'color' # 'gray' lub 'color'
    img_resize_shape = (128, 128)  # przykład: (32,32)

    base = f"./data/datasets/{data_split_type}/{img_type}/{img_resize_shape[0]}x{img_resize_shape[1]}"

    # wczytanie surowych danych (przed one-hot)
    train_set = load_array_from_file(f'{base}/train_set')
    val_set   = load_array_from_file(f'{base}/val_set')
    test_set  = load_array_from_file(f'{base}/test_set')

    # zliczenie próbek i klas w surowych zbiorach
    raw_sets = {
        'Train': train_set,
        'Validation': val_set,
        'Test': test_set
    }
    count_samples_and_classes(raw_sets)

    # przygotowanie X,y
    X_train, y_train = unpacking_data(train_set)
    X_val,   y_val   = unpacking_data(val_set)
    X_test,  y_test  = unpacking_data(test_set)

    # reshape dla MLP
    if neural_network == 'mlp':
        X_train = flatten_data(X_train, img_type)
        X_val   = flatten_data(X_val,   img_type)
        X_test  = flatten_data(X_test,  img_type)
        input_shape = (X_train.shape[1],)
    else:
        input_shape = (img_resize_shape[0], img_resize_shape[1], 1 if img_type=='gray' else 3)

    n_classes = y_train.shape[1]

    # tworzenie modelu
    if neural_network == 'mlp':
        model = create_model_mlp(n_classes, input_shape, optimizer='adam')
    else:
        model = create_model_cnn(n_classes, input_shape, optimizer='adam')

    # callback
    callbacks = [EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)]

    # trening i ewaluacja
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2000, batch_size=16, callbacks=callbacks)
    for name, (X_, y_) in [('Train', (X_train, y_train)), ('Val', (X_val, y_val)), ('Test', (X_test, y_test))]:
        loss, acc = model.evaluate(X_, y_, verbose=0)
        print(f"{name} accuracy: {acc*100:.2f}%")

if __name__ == '__main__':
    main()
