import argparse
import os


def split_dataset(source_path, train_path, test_path, train_ratio):
    """Splits the dataset on disk - creates a training and testing set from a single database
    """
    # Deleting train and test directories if they exist
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)

    # Creating directories
    os.makedirs(train_path)
    os.makedirs(test_path)

    classes = os.listdir(source_path)

    for c in classes:

        class_dir = os.path.join(source_path, c)

        images = os.listdir(class_dir)

        n_train = int(len(images) * train_ratio)

        train_images = images[:n_train]
        test_images = images[n_train:]

        os.makedirs(os.path.join(train_path, c), exist_ok=True)
        os.makedirs(os.path.join(test_path, c), exist_ok=True)

        for image in train_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(train_path, c, image)
            shutil.copyfile(image_src, image_dst)

        for image in test_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(test_path, c, image)
            shutil.copyfile(image_src, image_dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a dataset")
    parser.add_argument(
        "--source_path", default="data/images", help="Path to the dataset"
    )
    parser.add_argument(
        "--train_path",
        default="data/train",
        help="Path to the train set to be created",
    )
    parser.add_argument(
        "--test_path",
        default="data/test",
        help="Path to the test set to be created",
    )
    parser.add_argument(
        "--train_ratio",
        default=0.8,
        help="Train ratio - (dataset size / train set size)",
    )

    args = parser.parse_args()
    split_dataset(
        args.source_path, args.train_path, args.test_path, args.train_ratio
    )
