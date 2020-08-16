
import os
import argparse


def rename(path):
    """ Replaces the spaces, the dots and the colons with undersccores of a file or a directory (recursively)

    :param path: Path to the dataset
    :type path: str

    :rtype: None
    """
    new_path = path.replace(" ", "_").replace(".", "_").replace(":", "_")
    os.rename(path, new_path)
    path = new_path
    try:
        listdir = os.listdir(path)
    except NotADirectoryError:  # base case
        new_path = list(path)
        if new_path[-4] == "_":  # .JPG
            new_path[-4] = "."
        else:  # .JSON
            new_path[-5] = "."
        new_path = "".join(new_path)
        os.rename(path, new_path)
        return

    for dir in listdir:
        rename(os.path.join(path, dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename subfolders")
    parser.add_argument(
        "--path", default="data", help="Path to the dataset",
    )

    args = parser.parse_args()
    rename(args.path)
