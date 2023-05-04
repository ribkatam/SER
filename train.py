import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from downstream.solver import get_mockingjay_optimizer

from tqdm import tqdm
import yaml
from openpyxl import Workbook
import random
import numpy as np

from dataset import CustomDataset
from utils import EarlyStopper


def main():

    with open("./param.yaml", "r") as stream:
        params = yaml.safe_load(stream)

    model_params = params["model"]
    train_params = params["training"]
    batch_size = train_params["batch_size"]
    device = train_params["device"]
    exp_folder = train_params["exp_folder"]
    epoch = train_params["epoch"]
    lr = float(train_params["lr"])
    factor = float(train_params["scheduler"]["factor"])
    patience = float(train_params["scheduler"]["patience"])
    seed = int(train_params["seed"])
    train_path = train_params["train_path"]
    val_path = train_params["val_path"]
    worksheet_path = train_params["worksheet_path"]
    bert_conv_att = train_params["bert_conv_att"]
    fine_tune = train_params["fine_tune"]



    workbook = Workbook()
    worksheet = workbook.active
    worksheet["A1"] = "Epoch"
    worksheet["B1"] = "Training Loss"
    worksheet["C1"] = "Validation Loss"
    worksheet["D1"] = "Training Acc"
    worksheet["E1"] = "Validation Acc"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
   
    
    if bert_conv_att:
        from model import BERTConvAtt
        net = BERTConvAtt(model_params).to(device)
        if fine_tune:
            optimizer = torch.optim.Adam(list(net.parameters())+list(net.mj.mockingjay.parameters()), lr=lr)
            print("trainable parameters: {}".format(sum(p.numel() for p in net.parameters()) + sum(p.numel() for p in net.mj.mockingjay.parameters())))
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            print("trainable parameters: {}".format(sum(p.numel() for p in net.parameters())))
    else:
        from model import ConvAttBlock
        net = ConvAttBlock(model_params).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        print("trainable parameters: {}".format(sum(p.numel() for p in net.parameters())))

    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, verbose=True) 
    writer = SummaryWriter(exp_folder)

    dataset = CustomDataset(train_path, device=device)
    test_dataset = CustomDataset(val_path, device=device)
    

    dataloader =  DataLoader(dataset, batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size)


    early_stopper = EarlyStopper()
    best_acc = float("-inf")
    
  

    for e in tqdm(range(epoch)):
        # reset for training
        count = 0
        total_loss = 0
        total_acc =0
        train_loss = 0
        train_acc= 0
        net.train()
        for feature, label in dataloader:
          
            prediction = net(feature)
            loss = loss_fn(prediction, label)  # average loss of the batch          cur_batch_size = prediction.shape[0]  # the exact batch size

            cur_batch_size = prediction.shape[0]  # the exact batch size
            count += cur_batch_size  # accumulate over epoch
            total_loss +=loss* cur_batch_size  # accumulate loss over epoch
           
            total_acc +=(prediction.argmax(dim=1)== label).type(torch.float).sum()

            optimizer.zero_grad()
            loss.backward()  # back propagate every minibatch
            optimizer.step()
       
        # log every epoch
        train_loss = total_loss/count  # average over epoch
        train_acc= total_acc/count  

        writer.add_scalar('Loss/train', train_loss, e)
        writer.add_scalar("Acc/train", train_acc, e)


        # reset for validation
        count = 0
        total_loss = 0
        total_acc = 0
        val_loss = 0
        val_acc= 0

        with torch.no_grad():
            net.eval()
            for feature, label in test_dataloader:
                prediction = net(feature)
                loss = loss_fn(prediction, label)  # average loss of the batch          cur_batch_size = prediction.shape[0]  # the exact batch size

                cur_batch_size = prediction.shape[0]  # the exact batch size
                count += cur_batch_size  # accumulate over epoch
                total_loss +=loss* cur_batch_size  # accumulate loss over epoch
            
                total_acc +=(prediction.argmax(dim=1)== label).type(torch.float).sum()
                
            # log every epoch
            val_loss = total_loss/count # average over epoch
            val_acc= total_acc/count

            writer.add_scalar('Loss/val', val_loss , e)
            writer.add_scalar("Acc/val", val_acc, e)

            if val_acc > best_acc:
                best_acc = val_acc  
                torch.save(net.state_dict(), exp_folder + "/"+ "_best_acc.pt") # save the best validation acc state dict
                torch.save(best_acc, exp_folder+"/"+str(best_acc))
           
        print(best_acc)
        if early_stopper.early_stop(val_loss):
            break
        
        if(e % 100==0):
            torch.save(net.state_dict(), exp_folder + "/" + str(e) + ".pt")  # save every 100th epoch

        scheduler.step(val_loss)
          # write to an Excel file
        worksheet["A"+str(e+2)] = e+1
        worksheet["B"+str(e+2)] = train_loss.item()
        worksheet["C"+str(e+2)] = val_loss.item()
        worksheet["D"+str(e+2)] = train_acc.item()
        worksheet["E"+str(e+2)] = val_acc.item()
        workbook.save(worksheet_path)
    
if __name__ == "__main__":
    main()


