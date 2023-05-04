import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

from tqdm import tqdm
import os
import sys

from AudioFrame.model import Model
from AudioFrame.dataset import CustomDataset


        
def train_loop(model, loss_fn, optimizer, dataloader, device):
    total_loss = 0
    
    for frame, valence, arousal in dataloader:
        batch_prediction = model(frame.to(device).float())
        loss_valence = loss_fn(batch_prediction[:,0], valence.to(device).float())
        loss_arousal = loss_fn(batch_prediction[:,1], arousal.to(device).float())
        loss = 0.5 * (torch.abs(loss_valence) + torch.abs(loss_arousal))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+= loss.item()
        
    return total_loss/len(dataloader)


def test_loop(model, loss_fn, dataloader, device):
    total_loss = 0
   
    for frame, valence, arousal in dataloader:
        batch_prediction = model(frame.to(device).float())
        loss_valence = loss_fn(batch_prediction[:,0], valence.to(device).float())
        loss_arousal = loss_fn(batch_prediction[:,1], arousal.to(device).float())
        loss = 0.5 * (torch.abs(loss_valence) + torch.abs(loss_arousal))
        total_loss+= loss.item()
        
    return total_loss/len(dataloader)


def train(rank, world_size, loss_fn, lr, train_dataset, val_dataset, batch_size, epoch, experiment_path):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = "cuda:%d"%rank
    model = Model()
    model = DDP(model.to(device), device_ids=[rank])
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)


    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank) 
    val_sampler = DistributedSampler(dataset=val_dataset, num_replicas=world_size, rank=rank)   
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)


    if rank == 0:
        writer = SummaryWriter(experiment_path)
        best_val_loss = 1e4
        best_state_dict = model.state_dict()


    tqdm_file = sys.stderr if rank == 0 else open(os.devnull, 'w')
    pbar = tqdm(total=epoch, dynamic_ncols=True, file=tqdm_file)

    while pbar.n < pbar.total:

        train_sampler.set_epoch(pbar.n)
        val_sampler.set_epoch(pbar.n)

        train_loss = train_loop(model, loss_fn, optimizer, train_dataloader,  device)
        val_loss = test_loop(model, loss_fn, val_dataloader,  device)

        if rank == 0:
            writer.add_scalar('Loss/train', train_loss, pbar.n)
            writer.add_scalar("Loss/val", val_loss, pbar.n)
            if val_loss < best_val_loss:
                best_state_dict = model.state_dict()
                best_val_loss = val_loss

        pbar.update(1)
    
    if rank == 0:
        torch.save(best_state_dict, experiment_path + ".pt")
    
    dist.destroy_process_group()


def main():
   
    train_annotation_file = "/data1/ribka/Folders/AFF_WILD/AudioFrame/training_annotation.csv"
    val_annotation_file = "/data1/ribka/Folders/AFF_WILD/AudioFrame/validation_annotation.csv"
    experiment_path = "/data1/ribka/Folders/AFF_WILD/AudioFrame/Experiments/exp1_512_1e-4_sgd"
    train_dataset = CustomDataset(train_annotation_file)
    val_dataset = CustomDataset(val_annotation_file)
    

    # hyper parameters
    loss_fn = nn.MSELoss()
    lr = 1e-4
    batch_size = 256
    epoch = 1000
   
    
   
    world_size = 8
    mp.spawn(train,
        args=(world_size, loss_fn,lr, train_dataset, val_dataset, batch_size, epoch, experiment_path,),
        nprocs=world_size,
        join=True)

  
if __name__=="__main__":  
   
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
