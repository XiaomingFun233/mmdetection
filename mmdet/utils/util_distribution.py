# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

dp_factory = {'musa': MMDataParallel, 'cuda': MMDataParallel, 'cpu': MMDataParallel}

ddp_factory = {'musa': MMDistributedDataParallel, 'cuda': MMDistributedDataParallel}


def build_dp(model, device='musa', dim=0, *args, **kwargs):
    """build DataParallel module by device type.

    if device is musa, return a MMDataParallel model; if device is cuda,
    return a MMDataParallel model.

    Args:
        model (:class:`nn.Module`): model to be parallelized.
        device (str): device type, musa, cuda, cpu. Defaults to musa.
        dim (int): Dimension used to scatter the data. Defaults to 0.

    Returns:
        nn.Module: the model to be parallelized.
    """
    if device == 'npu':
        from mmcv.device.npu import NPUDataParallel
        dp_factory['npu'] = NPUDataParallel
        torch.npu.set_device(kwargs['device_ids'][0])
        torch.npu.set_compile_mode(jit_compile=False)
        model = model.npu()
    elif device == 'musa':
        model = model.musa(kwargs['device_ids'][0])
    elif device == 'cuda':
        model = model.cuda(kwargs['device_ids'][0])
    elif device == 'mlu':
        from mmcv.device.mlu import MLUDataParallel
        dp_factory['mlu'] = MLUDataParallel
        model = model.mlu()

    return dp_factory[device](model, dim=dim, *args, **kwargs)


def build_ddp(model, device='musa', *args, **kwargs):
    """Build DistributedDataParallel module by device type.

    If device is musa, return a MMDistributedDataParallel model;

    Args:
        model (:class:`nn.Module`): module to be parallelized.
        device (str): device type, musa or cuda.

    Returns:
        :class:`nn.Module`: the module to be parallelized

    References:
        .. [1] https://pytorch.org/docs/stable/generated/torch.nn.parallel.
                     DistributedDataParallel.html
    """
    assert device in ['musa', 'cuda', 'mlu',
                      'npu'], 'Only available for musa, cuda, mlu or npu devices.'
    if device == 'npu':
        from mmcv.device.npu import NPUDistributedDataParallel
        torch.npu.set_compile_mode(jit_compile=False)
        ddp_factory['npu'] = NPUDistributedDataParallel
        model = model.npu()
    elif device == 'musa':
        model = model.musa()
    elif device == 'cuda':
        model = model.cuda()
    elif device == 'mlu':
        from mmcv.device.mlu import MLUDistributedDataParallel
        ddp_factory['mlu'] = MLUDistributedDataParallel
        model = model.mlu()

    return ddp_factory[device](model, *args, **kwargs)


def is_npu_available():
    """Returns a bool indicating if NPU is currently available."""
    return hasattr(torch, 'npu') and torch.npu.is_available()


def is_mlu_available():
    """Returns a bool indicating if MLU is currently available."""
    return hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()


def is_musa_available():
    """Returns a bool indicating if MUSA is currently available."""
    return hasattr(torch, 'musa') and torch.musa.is_available()


def get_device():
    """Returns an available device, cpu, musa, cuda or mlu."""
    is_device_available = {
        'musa': is_musa_available(),
        'npu': is_npu_available(),
        'cuda': torch.cuda.is_available(),
        'mlu': is_mlu_available()
    }
    device_list = [k for k, v in is_device_available.items() if v]
    return device_list[0] if len(device_list) >= 1 else 'cpu'
