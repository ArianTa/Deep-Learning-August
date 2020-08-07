from collections import Counter
from numpy import log
import argparse
import os

def shannon_entropy(path):
    classes = os.listdir(path)
    # Counting the number of classes
    n = 0
    for kls in classes:
        kls_path = os.path.join(path, kls)
        n += len(os.listdir(kls_path))

    k = len(os.listdir(path))
    H = 0
    for kls in classes:
        kls_path = os.path.join(path, kls)
        c = len(os.listdir(kls_path))
        p = c / n
        H += - p * log(p) 

    return H / log(k)


if __name__ == "__main__":
    # Parsing arguments and setting up metadata
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--path",
        type=str,
        default="data/images",
        help="Path to root directory of the dataset",
    )
    # Add more stuff here maybe ?
    args = parser.parse_args()

    #Measure the balance of the dataset:
    print(f'shannon_entropy = {shannon_entropy(args.path)}')
