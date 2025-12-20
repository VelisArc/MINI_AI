import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os

class MultiGPUTrainer:
    """
    FIXED Multi-GPU trainer with proper DDP/FSDP support
    """
    def __init__(self, mode='ddp'):
        """
        mode: 'ddp' (DataParallel) or 'fsdp' (Fully Sharded)
        """
        self.mode = mode
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.is_distributed = self.world_size > 1
        
        if self.is_distributed:
            self._init_distributed()

    def _init_distributed(self):
        """Initialize distributed training"""
        print(f"[GPU-{self.rank}] Initializing distributed training...")
        
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',  # Use NCCL for GPU
                init_method='env://',
                world_size=self.world_size,
                rank=self.rank
            )
        
        torch.cuda.set_device(self.local_rank)
        print(f"[GPU-{self.rank}] Distributed init complete")

    def wrap_model(self, model):
        """Wrap model for multi-GPU"""
        if not self.is_distributed:
            return model.cuda() if torch.cuda.is_available() else model
        
        model = model.cuda(self.local_rank)
        
        if self.mode == 'fsdp':
            print(f"[GPU-{self.rank}] Using FSDP")
            model = FSDP(model)
        else:
            print(f"[GPU-{self.rank}] Using DDP")
            model = DDP(model, device_ids=[self.local_rank])
        
        return model

    def prepare_dataloader(self, dataset, batch_size, shuffle=True):
        """Create distributed dataloader"""
        if self.is_distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle
            )
            shuffle = False  # Sampler handles shuffling
        else:
            sampler = None
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=4
        )

    def is_main_process(self):
        """Check if this is the main process"""
        return self.rank == 0

    def save_checkpoint(self, model, path):
        """Save checkpoint (only from main process)"""
        if self.is_main_process():
            if self.is_distributed:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, path)
            print(f"[GPU-{self.rank}] Checkpoint saved: {path}")

    def cleanup(self):
        """Cleanup distributed training"""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()

