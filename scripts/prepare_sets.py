import os
import glob
import cv2
import random
import numpy as np


def get_images(path, ext='jpg'):
    img_files = glob.glob(f'{path}/*.{ext}')
    print(f"Loaded {len(img_files)} images.")
    return img_files


def save_array_to_file(file_path, data):
    # np.save will add “.npy” if it’s not already there
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    np.save(file_path, np.asarray(data))
    print(f">>> The array is saved in the file {file_path}.npy")


def load_array_from_file(file_path):
    # make sure we only append .npy once
    if not file_path.endswith('.npy'):
        file_path = f"{file_path}.npy"
    array = np.load(file_path, allow_pickle=True)
    return array


def gen_rand_indexes(num_range=(1, 101), n_num=2, iters=10):
    return [random.sample(range(num_range[0], num_range[1]), n_num) for _ in range(iters)]


def gen_sets_elem_indexes():
    rand_indexes = gen_rand_indexes(num_range=(0, 7), n_num=2, iters=25)
    rand_arr = np.array(rand_indexes)  # now shape (25,2)
    val_set_indexes  = rand_arr[:, 0]
    test_set_indexes = rand_arr[:, 1]

    save_array_to_file(file_path='./data/val_set_indexes',  data=val_set_indexes)
    save_array_to_file(file_path='./data/test_set_indexes', data=test_set_indexes)


def split_data_into_sets(array, indexes=None, sets_proportion=[5, 1, 1]):
    if indexes is not None:
        # `array` here is a 1-D object array; fancy-indexing works fine
        all_idxs = set(range(len(array)))
        diff_idx = list(all_idxs - set(indexes))

        train_set = list(array[diff_idx])
        if len(indexes) > 1:
            val_set  = [array[indexes[0]]]
            test_set = [array[indexes[1]]]
        else:
            test_set = [array[indexes[0]]]

    else:
        random.shuffle(array)
        arr = np.array(array, dtype=object)

        train_set = list(arr[:sets_proportion[0]])
        if len(sets_proportion) > 1:
            val_set  = list(arr[sets_proportion[0]:
                               sets_proportion[0]+sets_proportion[1]])
            test_set = list(arr[sets_proportion[0]+sets_proportion[1]:])
        else:
            test_set = list(arr[sets_proportion[0]:])

    if indexes is not None and len(indexes) > 1:
        return train_set, val_set, test_set
    else:
        return train_set, test_set


def prepare_sets(all_data, classes, indexes):
    global_train_set = []
    global_val_set   = []
    global_test_set  = []

    for idx, class_id in enumerate(classes):
        # force an object-dtype array so numpy won’t try to coerce shapes:
        single_class_set = np.array(
            [x for x in all_data if x[1] == class_id],
            dtype=object
        )

        train_set, val_set, test_set = split_data_into_sets(
            array=single_class_set,
            indexes=(indexes[0][idx], indexes[1][idx])
        )

        global_train_set.extend(train_set)
        global_val_set.extend(val_set)
        global_test_set.extend(test_set)

    print(f"Prepared sets: train={len(global_train_set)}, val={len(global_val_set)}, test={len(global_test_set)}")
    return global_train_set, global_val_set, global_test_set


def load_and_prepare_image(img_path, img_resize_shape, img_type, display_images=False):
    img = cv2.imread(img_path)
    if img_type == 'greyscale':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_resize_shape:
        img = cv2.resize(img, img_resize_shape)
    if display_images:
        cv2.imshow('Input', img); cv2.waitKey(0); cv2.destroyAllWindows()

    img = img.astype('float32') / 255.0
    return img


def create_dataset(img_list, img_resize_shape, img_type):
    dataset = []
    classes = []
    for img_path in img_list:
        image = load_and_prepare_image(img_path, img_resize_shape, img_type)
        class_idx = int(os.path.basename(img_path).split('_')[0])
        if class_idx not in classes:
            classes.append(class_idx)
        dataset.append([image, class_idx])
    return dataset, classes


def main():
    data_split_type = 'standard_split'
    img_type = 'greyscale'  # 'greyscale' / 'color'
    img_resize_shape = (32,32)

    if img_resize_shape:
        folder_to_save = f"./data/datasets/{data_split_type}/{img_type}/{img_resize_shape[0]}x{img_resize_shape[1]}"
    else:
        folder_to_save = f"./data/datasets/{data_split_type}/{img_type}/original_size"

    img_files = get_images(path='./data/subset-1', ext='jpg')
    dataset, classes = create_dataset(img_files, img_resize_shape, img_type)

    if data_split_type == 'standard_split':
        val_set_idxs  = load_array_from_file('./data/val_set_indexes')
        test_set_idxs = load_array_from_file('./data/test_set_indexes')

        train_set, val_set, test_set = prepare_sets(
            all_data=dataset,
            classes=classes,
            indexes=(val_set_idxs, test_set_idxs)
        )

        random.shuffle(train_set)
        random.shuffle(val_set)
        random.shuffle(test_set)

        # save them
        save_array_to_file(f'{folder_to_save}/train_set', train_set)
        save_array_to_file(f'{folder_to_save}/val_set',   val_set)
        save_array_to_file(f'{folder_to_save}/test_set',  test_set)

        # quick load-and-print check
        arr = load_array_from_file(f'{folder_to_save}/train_set')
        print(arr)

    else:
        random.shuffle(dataset)
        save_array_to_file(f'{folder_to_save}/dataset', dataset)
        arr = load_array_from_file(f'{folder_to_save}/dataset')
        print(arr)


if __name__ == "__main__":
    # gen_sets_elem_indexes()   # for task 1
    main()                     # for task 2
