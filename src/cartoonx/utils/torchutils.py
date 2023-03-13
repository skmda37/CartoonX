from typing import Union
import random
import numpy as np
import torch


def unnormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return x * std + mean


def check_nan(x: torch.Tensor) -> None:
    if torch.isnan(x).any():
        raise ValueError('Found NaN')


def init_cuda_device(devid: int) -> None:
    if devid is None:
        raise ValueError(
            'devid is None. Automatic cuda device selection not yet' /
            'implemented!'
        )
    else:
        torch.cuda.set_device(devid)

    print(f"torchutils.cuda_device_init device id = {devid}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def setup_gpu(devid: Union[int, None]):
    init_cuda_device(devid)


def init_seeds(seed: int = 1) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
