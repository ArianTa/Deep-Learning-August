import torch
import torchvision.transforms as transforms
import argparse
import torchvision.datasets as datasets


def stds_means(path):
    """ Computs the stds and means pf dataset
    
    :param path: Path to the dataset
    :type path: str

    :rtype: None
    """

    train_data = datasets.ImageFolder(
        root=path, transform=transforms.ToTensor(),
    )

    means = torch.zeros(3)
    stds = torch.zeros(3)

    for (img, label,) in train_data:
        means += torch.mean(img, dim=(1, 2,),)
        stds += torch.std(img, dim=(1, 2,),)

    means /= len(train_data)
    stds /= len(train_data)
    return (
        means,
        stds,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute means and stds of a dataset"
    )
    parser.add_argument(
        "--source_path", default="data/images", help="Path to the dataset",
    )
    args = parser.parse_args()
    (means, stds,) = stds_means(args.source_path)

    print(f"Calculated means: {means}")
    print(f"Calculated stds: {stds}")
