import os
import torch.distributed as torch_distrib
from pytorch_lightning import _logger as log
import datetime
import torch
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.overrides import LightningDistributedModule
from torch.nn.parallel.distributed import DistributedDataParallel
from typing import Optional

from pytorch_lightning.utilities.distributed import rank_zero_info

if os.name == 'posix':
    import resource


def increase_filedesc_limit(n=4096):
    if os.name == 'posix':
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (n, rlimit[1]))


class MyDDP(DDPPlugin):
    """
     enable timeout
    """

    def init_ddp_connection(self, global_rank: Optional[int] = None, world_size: Optional[int] = None) -> None:
        global_rank = global_rank if global_rank is not None else self.cluster_environment.global_rank()
        world_size = world_size if world_size is not None else self.cluster_environment.world_size()
        os.environ["MASTER_ADDR"] = self.cluster_environment.master_address()
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        os.environ['NCCL_BLOCKING_WAIT'] = '1'

        if torch.distributed.is_available() and not torch.distributed.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            torch.distributed.init_process_group(
                self.torch_distributed_backend, rank=global_rank, world_size=world_size, timeout=datetime.timedelta(minutes=15)
            )

            # on rank=0 let everyone know training is starting
            rank_zero_info(
                f"{'-' * 100}\n"
                f"distributed_backend={self.torch_distributed_backend}\n"
                f"All DDP processes registered. Starting ddp with {self.world_size} processes\n"
                f"{'-' * 100}\n"
            )

