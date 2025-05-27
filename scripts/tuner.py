import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import ParameterGrid

# Aliasy dla warstw
EarlyStopping = tf.keras.callbacks.EarlyStopping
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
np_utils = tf.keras.utils


def load_array_from_file(file_name):
    """Wczytuje tablicę NumPy z pliku .npy"""
    if not file_name.endswith('.npy'):
        file_name += '.npy'
    return np.load(file_name, allow_pickle=True)


def load_dataset(main_path_sets):
    """Wczytuje zestawy: train, val, test z katalogu main_path_sets"""
    train_set = load_array_from_file(f'{main_path_sets}/train_set')
    val_set   = load_array_from_file(f'{main_path_sets}/val_set')
    test_set  = load_array_from_file(f'{main_path_sets}/test_set')
    return train_set, val_set, test_set


def unpacking_data(data):
    """Rozpakowuje X i y, konwertuje X na float32, y na kategorie"""
    X = data[:, 0]
    y = data[:, 1]
    X = np.array(list(X), dtype=np.float32)
    y = np_utils.to_categorical(y)
    return X, y


def flatten_data(dataset, img_type):
    """Spłaszcza obrazy do wektora"""
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
    """Tworzy i kompiluje MLP"""
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
    """Tworzy i kompiluje CNN"""
    model = Sequential([
        Conv2D(32, (3,3), padding='same',
               activation=func_activation,
               kernel_initializer=kernel_initializer,
               input_shape=input_shape),
        Conv2D(32, (3,3), padding='same',
               activation=func_activation,
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


def tune_and_log(param_grid,
                 X_train, y_train,
                 X_val,   y_val,
                 X_test,  y_test,
                 img_type,
                 results_csv_path='hyperparam_results.csv'):
    """
    Przegląd siatki hiperparametrów, trenowanie modeli i zapisywanie wyników do CSV.
    """
    results = []

    for params in ParameterGrid(param_grid):
        net = params['network']
        optimizer = params['optimizer']
        activation = params['activation']
        kernel_init = params['kernel_initializer']
        batch_size = params['batch_size']
        epochs = params['epochs']

        # Przygotowanie danych i modelu
        if net == 'mlp':
            X_tr = flatten_data(X_train, img_type)
            X_vl = flatten_data(X_val,   img_type)
            X_te = flatten_data(X_test,  img_type)
            model = create_model_mlp(
                n_classes=y_train.shape[1],
                input_shape=(X_tr.shape[1],),
                optimizer=optimizer,
                func_activation=activation,
                kernel_initializer=kernel_init
            )
        else:
            X_tr, X_vl, X_te = X_train, X_val, X_test
            model = create_model_cnn(
                n_classes=y_train.shape[1],
                input_shape=X_tr.shape[1:],
                optimizer=optimizer,
                func_activation=activation,
                kernel_initializer=kernel_init
            )

        # Callback EarlyStopping
        callbacks = [EarlyStopping(monitor='val_accuracy',
                                   patience=20,
                                   restore_best_weights=True)]

        # Trenowanie
        model.fit(
            X_tr, y_train,
            validation_data=(X_vl, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

        # Ewaluacja
        loss, acc = model.evaluate(X_te, y_test, verbose=0)
        acc_pct = round(acc * 100, 2)

        # Zapis wyników
        res = params.copy()
        res['test_accuracy_%'] = acc_pct
        results.append(res)
        print(f"Params: {params} -> Test acc: {acc_pct}%")

    # Eksport do CSV
    df = pd.DataFrame(results)
    df.to_csv(results_csv_path, index=False)
    print(f"Wszystkie wyniki zapisane w {results_csv_path}")
    return df


def main():
    # Ustawienia ścieżki do danych
    data_split_type = 'standard_split'
    img_type = 'color'
    img_size = (128,128)
    base = f"./data/datasets/{data_split_type}/{img_type}/{img_size[0]}x{img_size[1]}"

    # Wczytanie danych
    train_set, val_set, test_set = load_dataset(base)
    X_train, y_train = unpacking_data(train_set)
    X_val,   y_val   = unpacking_data(val_set)
    X_test,  y_test  = unpacking_data(test_set)

    # Definicja siatki hiperparametrów
    param_grid = {
        'network': ['mlp', 'cnn'],
        'optimizer': ['adam', 'rmsprop'],
        'activation': ['relu', 'tanh'],
        'kernel_initializer': ['random_uniform', 'he_uniform'],
        'batch_size': [16, 32],
        'epochs': [50, 100]
    }

    # Uruchom strojenie
    tune_and_log(
        param_grid,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        img_type,
        results_csv_path='hyperparam_results.csv'
    )

if __name__ == '__main__':
    main()
