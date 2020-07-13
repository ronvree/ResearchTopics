
import torch


def batch_tensors(*ts) -> torch.Tensor:
    """
    Convert multiple tensors of the same shape to a single batch tensor
    The batch size is the length of xs
    :param ts: the tensors that should be batched
    :return: a single batch tensor containing the individual tensors
    """
    return torch.cat([t.unsqueeze(0) for t in ts], dim=0)

