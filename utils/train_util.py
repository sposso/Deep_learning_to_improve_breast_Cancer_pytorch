from typing import List, Tuple
import os
import io
import shutil
from contextlib import contextmanager

import numpy
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torchvision.transforms as T
from utils.tools import CBIS_MAMMOGRAM,MyIntensityShift
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from utils.classifier import Bottleneck, Resnet50



def first_stage(model):
    
    #print("Params to update in the first stage ")
    params_to_update_first = []
    for name, param in model.named_parameters():
        if name.startswith('module.layer') or name.startswith("module.fc"):
            param.requires_grad = True
            params_to_update_first.append(param)
            #print("\t",name)
        else:
            param.requires_grad = False
    first_optimizer = optim.Adam(params_to_update_first, lr = 1e-4, weight_decay=1e-4)
    
    return first_optimizer

def second_stage(model):


    #print("Params to update in the second stage")
    params_to_update_second = []
    for name, param in model.named_parameters():

        param.requires_grad = True
        params_to_update_second.append(param)
        #print("\t",name)

    second_optimizer = optim.Adam(params_to_update_second, lr = 1e-5, weight_decay=0.001)
    
    return second_optimizer



def initialize_data_loader(batch_size,workers,root,aug = False) -> Tuple[DataLoader, DataLoader, DataLoader]:

    
    train = os.path.join(root,"data/train.csv")
    validation =os.path.join(root,"data/validation.csv")
    test = os.path.join(root,"data/test.csv")
   

    normalize = T.Normalize(mean=[0.2030],
                                     std=[0.2646])

    augmentation = T.Compose([normalize,T.RandomHorizontalFlip(), T.RandomVerticalFlip(),T.RandomRotation(degrees=25),
                        T.RandomAffine(degrees=0, scale=(0.8, 0.99)),T.RandomResizedCrop(size=(1152,896),scale=(0.8,0.99)),
                        MyIntensityShift(shift= [80,120]), T.RandomAffine(degrees=0, shear=12)])


    if aug:
        train_dataset= CBIS_MAMMOGRAM(train, transform = augmentation)
    
    else: 
        train_dataset = CBIS_MAMMOGRAM(train, transform = T.Compose([T.ToTensor(),normalize]))
    #Normalizing the validation set
    validation_dataset = CBIS_MAMMOGRAM(validation, transform = T.Compose([T.ToTensor(),normalize]))
    #Normalizing the test set
    test_dataset = CBIS_MAMMOGRAM(test, transform = T.Compose([T.ToTensor(),normalize]))

     # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = ElasticDistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)


    return train_loader,val_loader,test_loader

def Initialize_model(device_id,root):
    model = Resnet50(Bottleneck, layers=[2,2], use_pretrained=True, root=root)
    model.cuda(device_id)
    cudnn.benchmark = True
    model = DistributedDataParallel(model, device_ids=[device_id],find_unused_parameters=False)

    criterion = nn.CrossEntropyLoss().cuda(device_id)

    return model, criterion
   

class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """

    def __init__(self, model, optimizer):
        self.epoch = -1
        self.best_acc1 = 0
        self.model = model
        self.optimizer = optimizer

    def capture_snapshot(self):
        """
        Essentially a ``serialize()`` function, returns the state as an
        object compatible with ``torch.save()``. The following should work
        ::
        snapshot = state_0.capture_snapshot()
        state_1.apply_snapshot(snapshot)
        assert state_0 == state_1
        """
        return {
            "epoch": self.epoch,
            "best_acc1": self.best_acc1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def apply_snapshot(self, obj, device_id):
        """
        The complimentary function of ``capture_snapshot()``. Applies the
        snapshot object that was returned by ``capture_snapshot()``.
        This function mutates this state object.
        """

        self.epoch = obj["epoch"]
        self.best_acc1 = obj["best_acc1"]
        self.state_dict = obj["state_dict"]
        self.model.load_state_dict(obj["state_dict"])
        self.optimizer.load_state_dict(obj["optimizer"])

    def save(self, f):
        torch.save(self.capture_snapshot(), f)

    def load(self, f, device_id):
        # Map model to be loaded to specified single gpu.
        snapshot = torch.load(f, map_location=f"cuda:{device_id}")
        self.apply_snapshot(snapshot, device_id)


def load_checkpoint(
    checkpoint_file: str,
    device_id: int,
    model: DistributedDataParallel,
    optimizer,  # SGD
) -> State:
    """
    Loads a local checkpoint (if any). Otherwise, checks to see if any of
    the neighbors have a non-zero state. If so, restore the state
    from the rank that has the most up-to-date checkpoint.
    .. note:: when your job has access to a globally visible persistent storage
              (e.g. nfs mount, S3) you can simply have all workers load
              from the most recent checkpoint from such storage. Since this
              example is expected to run on vanilla hosts (with no shared
              storage) the checkpoints are written to local disk, hence
              we have the extra logic to broadcast the checkpoint from a
              surviving node.
    """

    state = State( model, optimizer)

    if os.path.isfile(checkpoint_file):
        print(f"=> loading checkpoint file: {checkpoint_file}")
        state.load(checkpoint_file, device_id)
        print(f"=> loaded checkpoint file: {checkpoint_file}")

    # logic below is unnecessary when the checkpoint is visible on all nodes!
    # create a temporary cpu pg to broadcast most up-to-date checkpoint
    with tmp_process_group(backend="gloo") as pg:
        rank = dist.get_rank(group=pg)

        # get rank that has the largest state.epoch
        epochs = torch.zeros(dist.get_world_size(), dtype=torch.int32)
        epochs[rank] = state.epoch
        dist.all_reduce(epochs, op=dist.ReduceOp.SUM, group=pg)
        t_max_epoch, t_max_rank = torch.max(epochs, dim=0)
        max_epoch = t_max_epoch.item()
        max_rank = t_max_rank.item()

        # max_epoch == -1 means no one has checkpointed return base state
        if max_epoch == -1:
            print(f"=> no workers have checkpoints, starting from epoch 0")
            return state

        # broadcast the state from max_rank (which has the most up-to-date state)
        # pickle the snapshot, convert it into a byte-blob tensor
        # then broadcast it, unpickle it and apply the snapshot
        print(f"=> using checkpoint from rank: {max_rank}, max_epoch: {max_epoch}")

        with io.BytesIO() as f:
            torch.save(state.capture_snapshot(), f)
            raw_blob = numpy.frombuffer(f.getvalue(), dtype=numpy.uint8)

        blob_len = torch.tensor(len(raw_blob))
        dist.broadcast(blob_len, src=max_rank, group=pg)
        print(f"=> checkpoint broadcast size is: {blob_len}")

        if rank != max_rank:
            # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
            #  typing.Tuple[int, ...]]` but got `Union[bool, float, int]`.
            blob = torch.zeros(blob_len.item(), dtype=torch.uint8)
        else:
            blob = torch.as_tensor(raw_blob, dtype=torch.uint8)

        dist.broadcast(blob, src=max_rank, group=pg)
        print(f"=> done broadcasting checkpoint")

        if rank != max_rank:
            with io.BytesIO(blob.numpy()) as f:
                snapshot = torch.load(f)
            state.apply_snapshot(snapshot, device_id)

        # wait till everyone has loaded the checkpoint
        dist.barrier(group=pg)

    print(f"=> done restoring from previous checkpoint")
    return state

@contextmanager
def tmp_process_group(backend):
    cpu_pg = dist.new_group(backend=backend)
    try:
        yield cpu_pg
    finally:
        dist.destroy_process_group(cpu_pg)


def save_checkpoint(state: State, is_best: bool, filename: str):
    checkpoint_dir = os.path.dirname(filename) #
    os.makedirs(checkpoint_dir, exist_ok=True) #Create directory

    # save to tmp, then commit by moving the file in case the job
    # gets interrupted while writing the checkpoint
    tmp_filename = filename + ".tmp"
    torch.save(state.capture_snapshot(), tmp_filename)
    os.rename(tmp_filename, filename)
    print(f"=> saved checkpoint for epoch {state.epoch} at {filename}")
    if is_best:
        best = os.path.join(checkpoint_dir, "model_best.pth.tar")
        print(f"=> best model found at epoch {state.epoch} saving to {best}")
        shutil.copyfile(filename, best)


