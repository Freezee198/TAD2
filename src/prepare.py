import os
import numpy as np
import sys
import pickle


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-folder\n")
        sys.exit(1)

    data_folder = sys.argv[1]
    train_dir = data_folder + '/train'
    val_dir = data_folder + '/valid'
    test_dir = data_folder + '/test'
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

    os.makedirs(os.path.join("prepared"), exist_ok=True)

    np.save("prepared/trainX", trainX)
    np.save("prepared/trainY", trainY)
    np.save("prepared/valX", valX)
    np.save("prepared/valY", valY)
    np.save("prepared/testX", testX)
    np.save("prepared/testY", testY)
    with open("prepared/CLASS NAMES", "wb") as fp:
        pickle.dump(class_names, fp)


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
