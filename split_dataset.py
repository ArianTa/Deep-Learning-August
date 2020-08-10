import argparse
import os
import shutil
import json
import random

if __name__ == "__main__": #4795
    parser = argparse.ArgumentParser(description="Split a dataset")
    parser.add_argument(
        "--train_json", default="data/train.json", help="Path to the dataset",
    )
    parser.add_argument(
        "--test_json",
        default="data/val.json",
        help="Path to the train set to be created",
    )
    args = parser.parse_args()

    with open(args.train_json) as json_file:
        train_annotation = json.load(json_file)

    with open(args.test_json) as json_file:
        test_annotation = json.load(json_file)


    for i in range(4795):
        idx = random.randint(0, len(train_annotation['annotations']))
        annotation = train_annotation['annotations'][idx]
        del train_annotation['annotations'][idx]
        image = train_annotation['images'][idx]
        del train_annotation['images'][idx]

        test_annotation['annotations'].append(annotation)
        test_annotation['images'].append(image)


    with open('train.json', 'w') as outfile:
        json.dump(train_annotation, outfile)

    with open('test.json', 'w') as outfile:
        json.dump(test_annotation, outfile)
