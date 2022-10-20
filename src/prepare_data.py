import os
import numpy as np


def main():
    train_dir = '../data/train'
    val_dir = '../data/valid'
    test_dir = '../data/test'
    class_names = os.listdir(train_dir)

    train_image_files = [[os.path.join(train_dir, class_name, x)
                          for x in os.listdir(os.path.join(train_dir, class_name))]
                         for class_name in class_names]

    valid_image_files = [[os.path.join(val_dir, class_name, x)
                          for x in os.listdir(os.path.join(val_dir, class_name))]
                         for class_name in class_names]

    test_image_files = [[os.path.join(test_dir, class_name, x)
                         for x in os.listdir(os.path.join(test_dir, class_name))]
                        for class_name in class_names]

    train_file_list, train_label_list = collect_images(train_image_files, class_names)
    valid_file_list, valid_label_list = collect_images(valid_image_files, class_names)
    test_file_list, test_label_list = collect_images(test_image_files, class_names)

    trainX = np.array(train_file_list)
    trainY = np.array(train_label_list)
    valX = np.array(valid_file_list)
    valY = np.array(valid_label_list)
    testX = np.array(test_file_list)
    testY = np.array(test_label_list)

    np.save("../data/processed_data/trainX", trainX)
    np.save("../data/processed_data/trainY", trainY)
    np.save("../data/processed_data/valX", valX)
    np.save("../data/processed_data/valY", valY)
    np.save("../data/processed_data/testX", testX)
    np.save("../data/processed_data/testY", testY)


def collect_images(image_files, class_names):
    image_file_list0 = []
    image_label_list0 = []
    for i, class_name in enumerate(class_names):
        image_file_list0.extend(image_files[i])
        image_label_list0.extend([i] * len(image_files[i]))
    image_file_list = []
    image_label_list = []
    for i, path in enumerate(image_file_list0):
        IMG0 = path.split('.')[-1]
        if IMG0 != 'svg' and IMG0 != 'gif':
            image_file_list += [path]
            image_label_list += [image_label_list0[i]]
    return image_file_list, image_label_list


if __name__ == "__main__":
    main()
