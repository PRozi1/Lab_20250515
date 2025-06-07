import os
import numpy as np
import tensorflow as tf

from datetime import datetime
from sklearn.model_selection import StratifiedKFold, KFold
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


def load_dataset_cv(main_path_sets):
    return load_array_from_file(f'{main_path_sets}/dataset')


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


def gen_test_folder_name():
    now = datetime.now()
    year = now.year
    month = f"{now.month:02d}"
    day   = f"{now.day:02d}"
    hour  = f"{now.hour:02d}"
    minute= f"{now.minute:02d}"
    return f"test_{year}{month}{day}_{hour}{minute}"


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


def reshape_data(dataset):
    # TODO: implement if you want to support CNN->MLP reshape
    raise NotImplementedError()


def train_evaluate(X_train, y_train, X_test, y_test, model, epochs, batch_size, callbacks):
    # y_train = np_utils.to_categorical(y_train)
    # y_test  = np_utils.to_categorical(y_test)
    n_classes = model.output_shape[-1]
    y_train = np_utils.to_categorical(y_train, num_classes=n_classes)
    y_test  = np_utils.to_categorical(y_test,  num_classes=n_classes)
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size,
              callbacks=callbacks)
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    return round(acc * 100, 2)


# def cross_validation(n_splits, X, y, model, epochs, batch_size, callbacks):
#     kf = StratifiedKFold(n_splits=n_splits)
#     y_labels = np.argmax(y, axis=1)

def cross_validation(n_splits, X, y, model, epochs, batch_size, callbacks):
    kf = StratifiedKFold(n_splits=n_splits)
    # jeśli y ma więcej wymiarów, to to-hot → zamień na etykiety, w przeciwnym razie już etykiety
    # if y.ndim > 1:
    #     y_labels = np.argmax(y, axis=1)
    # else:
    #     y_labels = y
    if y.ndim > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y
    # jeśli nie da się stratify (np. min_count < n_splits), użyj zwykłego KFold
    from collections import Counter
    min_count = min(Counter(y_labels).values())
    if min_count < n_splits:
        print(f"Uwaga: klasa z najmniejszą liczbą próbek ma tylko {min_count} < {n_splits}, przełączam na KFold")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = splitter.split(X)
    else:
        splitter = StratifiedKFold(n_splits=n_splits)
        splits = splitter.split(X, y_labels)
    test_scores = []

    # save initial weights (plik musi kończyć się na .weights.h5)
    #model.save_weights('reference_model.weights.h5')
    # save initial weights (plik musi kończyć się na .weights.h5)
    model.save_weights('reference_model.weights.h5')
    
    print(f"{n_splits}-Fold Cross Validation\n")

    #for idx, (train_idx, test_idx) in enumerate(kf.split(X, y_labels), start=1):
    for idx, (train_idx, test_idx) in enumerate(splits, start=1):
        # reload initial weights before each fold
        model.load_weights('reference_model.weights.h5')
        score = train_evaluate(
            X[train_idx], y_labels[train_idx],
            X[test_idx],  y_labels[test_idx],
            model, epochs, batch_size, callbacks
        )
        test_scores.append(score)
        print(f"Fold {idx}: test acc = {score}%")

    mean = round(np.mean(test_scores), 2)
    std  = round(np.std(test_scores), 2)
    print(f"\nFinal results: mean={mean}%, std={std}%")
    return test_scores

# def cross_validation(n_splits, X, y, model, epochs, batch_size, callbacks):
#     kf = StratifiedKFold(n_splits=n_splits)
#     y_labels = np.argmax(y, axis=1)
#     test_scores = []

#     # save initial weights
#     model.save_weights('reference_model.h5')
#     print(f"{n_splits}-Fold Cross Validation\n")

#     for idx, (train_idx, test_idx) in enumerate(kf.split(X, y_labels), start=1):
#         model.load_weights('reference_model.h5')
#         score = train_evaluate(X[train_idx], y_labels[train_idx],
#                                X[test_idx],  y_labels[test_idx],
#                                model, epochs, batch_size, callbacks)
#         test_scores.append(score)
#         print(f"Fold {idx}: test acc = {score}%")

#     mean = round(np.mean(test_scores), 2)
#     std  = round(np.std(test_scores), 2)
#     print(f"\nFinal results: mean={mean}%, std={std}%")
#     return test_scores


def create_model_mlp(n_classes, input_shape,
                     optimizer='rmsprop', func_activation='relu',
                     kernel_initializer='random_uniform'):
    model = Sequential([
        Dense(128, input_shape=input_shape,
              activation=func_activation,
               kernel_initializer=kernel_initializer),
        #  Dense(128, activation=func_activation,
        #        kernel_initializer=kernel_initializer),
        #  Dense(256, activation=func_activation,
        #        kernel_initializer=kernel_initializer),
        Dense(n_classes, activation='softmax',
              kernel_initializer='random_uniform'),
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()
    return model


def create_model_cnn(n_classes, input_shape,
                     optimizer='adam', func_activation='relu',
                     kernel_initializer='random_uniform'):
    model = Sequential([
        Conv2D(32, (3, 3), padding='same',
               activation=func_activation,
               kernel_initializer=kernel_initializer,
               input_shape=input_shape),
        Conv2D(32, (3, 3), padding='same',
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
    model.summary()
    return model


def main():
    data_split_type      = 'standard_split'  # or 'cv'
    neural_network       = 'mlp'             # or 'cnn'
    img_type             = 'color'       # or 'color'
    img_resize_shape     = (128,128)              # or (X, Y)
    reshape_data_method  = 'data_flattening' # or 'data_reshape'

    # -- BUILD the path --
    base = f"./data/datasets/{data_split_type}/{img_type}"
    if img_resize_shape:
        base = f"{base}/{img_resize_shape[0]}x{img_resize_shape[1]}"
    else:
        base = f"{base}/original_size"        # <--- fixed spelling!

    # create logs folder
    test_folder = gen_test_folder_name()
    os.makedirs(f"./logs/{test_folder}", exist_ok=True)

    # load data
    if data_split_type == 'standard_split':
        train_set, val_set, test_set = load_dataset(base)
        X_train, y_train = unpacking_data(train_set)
        X_val,   y_val   = unpacking_data(val_set)
        X_test,  y_test  = unpacking_data(test_set)
    else:
        dataset = load_dataset_cv(base)
        X, y = unpacking_data(dataset)

    # determine shapes
    if data_split_type == 'standard_split':
        channels = 1 if img_type == 'greyscale' else X_train.shape[-1]
        img_shape = (X_train.shape[1], X_train.shape[2], channels)
        n_classes = y_train.shape[1]
    else:
        channels = 1 if img_type == 'greyscale' else X.shape[-1]
        img_shape = (X.shape[1], X.shape[2], channels)
        n_classes = y.shape[1]

    # flatten if MLP
    if neural_network == 'mlp':
        if reshape_data_method == 'data_flattening':
            if data_split_type == 'standard_split':
                X_train = flatten_data(X_train, img_type)
                X_val   = flatten_data(X_val,   img_type)
                X_test  = flatten_data(X_test,  img_type)
            else:
                X = flatten_data(X, img_type)
        elif reshape_data_method == 'data_reshape':
            # implement reshape_data if you need it
            raise NotImplementedError()

        model = create_model_mlp(
            n_classes=n_classes,
            input_shape=(X_train.shape[1],),
            optimizer='adam'
        )
    else:
        model = create_model_cnn(
            n_classes=n_classes,
            input_shape=img_shape,
            optimizer='adam'
        )

    # callbacks
    callbacks = [
        EarlyStopping(monitor="val_accuracy",
                      patience=100,
                      restore_best_weights=True)
    ]

    # training / evaluation
    if data_split_type == 'standard_split':
        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=2000, batch_size=16,
                  callbacks=callbacks)

        for name, (X_, y_) in [('Train', (X_train, y_train)),
                               ('Val',   (X_val,   y_val)),
                               ('Test',  (X_test,  y_test))]:
            loss, acc = model.evaluate(X_, y_, verbose=0)
            print(f"{name} acc: {acc*100:.2f}%")
    else:
        cross_validation(
            n_splits=30, X=X, y=y,
            model=model,
            epochs=2000, batch_size=32,
            callbacks=callbacks
        )


if __name__ == "__main__":
    main()
