from .testcnn import TestCNN
from .resnet import *
from .vgg import *
from .densenet import *
from .googlenet import *
from .shufflenetV2 import *
from .alexnet import *
from .misc import *



def get_model(args):
    import os
    output_dim = len(os.listdir(args.path))
    if "resnet" in args.model:
        (model, params) = get_resnet_model(args.model, args.lr, output_dim)
    elif "vgg" in args.model:
        (model, params) = get_vgg_model(args.model, args.lr, output_dim)
    elif "googlenet" in args.model:
        (model, params) = get_googlenet_model(args.lr, output_dim)
    elif "densenet" in args.model:
        (model, params) = get_densenet_model(args.model, args.lr, output_dim)
    elif "shufflenet" in args.model:
        (model, params) = get_shufflenet_model(
            args.model, args.lr, output_dim,
        )
    elif "alexnet" in args.model:
        (model, params) = get_alexnet_model(args.model, args.lr, output_dim)
    else:
        (model, params) = get_all_model(args.model, args.lr, output_dim)

    return model, params
