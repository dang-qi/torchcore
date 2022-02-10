import math
from typing import TypeVar, Optional, Iterator, Sized

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
import numpy as np


T_co = TypeVar('T_co', covariant=True)

from .build import SAMPLER_REG

@SAMPLER_REG.register()
class GroupSampler(Sampler[T_co]):
    def __init__(self, data_source: Optional[Sized], num_per_gpu: int, shuffle: bool=False, num_replicas=None, rank=None, seed: int=0) -> None:
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        if dist.is_available() and dist.is_initialized():
            self.distributed = True
        else:
            self.distributed = False
        self.epoch = 0
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        #print('group sampler distributed/replicas/rank:',self.distributed, num_replicas, rank)

        self.dataset = data_source
        self.num_per_gpu = num_per_gpu
        self.shuffle = shuffle
        self.seed = seed
        self.group_flag= data_source.aspect_ratio_flag
        self.group_size = np.bincount(self.group_flag)
        self.num_replicas = num_replicas

        self.num_gpu_group = 0
        for size in self.group_size:
            self.num_gpu_group += math.ceil(size/ num_per_gpu)  # type: ignore

        self.num_samples = math.ceil(self.num_gpu_group / self.num_replicas) * self.num_per_gpu
        self.total_size = math.ceil(self.num_gpu_group / self.num_replicas) * self.num_replicas * num_per_gpu

        self.world_batch_size = self.num_per_gpu * self.num_replicas

    def __iter__(self) -> Iterator[T_co]:
        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        if self.distributed:
            g.manual_seed(self.seed + self.epoch)
        indices = []
        for i, size in enumerate(self.group_size):
            if size>0:
                inds = np.where(self.group_flag==i)[0]
                assert len(inds) == size
                if self.shuffle:
                    rand_ind = torch.randperm(size, generator=g)
                    inds = inds[rand_ind.numpy()]
                inds = inds.tolist()
                if size % self.num_per_gpu != 0:
                    # in case the size is less than num per gpu
                    for _ in range(self.num_per_gpu//size-1):
                        inds.extend(inds)

                    pad = self.num_per_gpu - len(inds)%self.num_per_gpu
                    inds.extend(inds[:pad])
                indices.append(torch.tensor(inds).reshape((-1,self.num_per_gpu)))

        indices = torch.cat(indices, dim=0)
        if self.shuffle:
            shuffle_ind = torch.randperm(len(indices), generator=g)
            indices = indices[shuffle_ind]

        if len(indices) % self.num_replicas != 0:
            if self.num_replicas > len(indices):
                indices.repeat((self.num_replicas//len(indices),1))
            pad = self.num_replicas - len(indices)%self.num_replicas
            indices = torch.cat((indices, indices[:pad]), dim=0)

        if self.distributed:
            # to adapt the style of DistributedSampler
            indices = indices.reshape((-1,self.num_replicas, self.num_per_gpu)).permute(0,2,1)

        indices = indices.flatten().tolist()

        return iter(indices)

    def __len__(self) -> int:
        return self.total_size

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch



# revise from torch DistributedSampler
class DistributedGroupSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_per_gpu (int): Number of samples in each gpu
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_per_gpu: int, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0,) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.num_per_gpu = num_per_gpu
        self.rank = rank
        self.epoch = 0
        self.group_flag = dataset.aspect_ratio_flag
        self.group_size = np.bincount(self.group_flag)
        self.world_batch_size = self.num_per_gpu * self.num_replicas

        self.num_gpu_group = 0
        for i, size in enumerate(self.group_size):
            self.num_gpu_group += math.ceil(size/ num_per_gpu)  # type: ignore

        self.num_samples = math.ceil(self.num_gpu_group / self.num_replicas) * self.num_per_gpu
        self.total_size = math.ceil(self.num_gpu_group / self.num_replicas) * self.num_replicas * num_per_gpu

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = []
        for i, size in enumerate(self.group_size):
            if size>0:
                inds = np.where(self.group_flag==i)[0]
                assert len(inds) == size
                if self.shuffle:
                    rand_ind = torch.randperm(size, generator=g)
                    inds = inds[rand_ind.numpy()]
                inds = inds.tolist()
                if size % self.num_per_gpu != 0:
                    # in case the size is less than num per gpu
                    for _ in range(self.num_per_gpu//size-1):
                        inds.extend(inds)

                    pad = self.num_per_gpu - len(inds)%self.num_per_gpu
                    inds.extend(inds[:pad])
                indices.append(torch.tensor(inds).reshape((-1,self.num_per_gpu)))

        indices = torch.cat(indices, dim=0)
        if self.shuffle:
            shuffle_ind = torch.randperm(len(indices), generator=g)
            indices = indices[shuffle_ind]

        if len(indices) % self.num_replicas != 0:
            if self.num_replicas > len(indices):
                indices.repeat((self.num_replicas//len(indices),1))
            pad = self.num_replicas - len(indices)%self.num_replicas
            indices = torch.cat((indices, indices[:pad]), dim=0)

        indices = indices.flatten().tolist()

        assert len(indices) == self.total_size

        # subsample
        offset = self.rank * self.num_samples
        indices = indices[offset: offset+self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
