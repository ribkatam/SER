import torch
import torch.nn as nn
import openpyxl

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tqdm import tqdm
from dataset import CustomDataset
from utils import EarlyStopper
 
from features_new3 import MFCC, MFCCDerivatives
from model_new3 import Net3, Net3_att, Net3_att_mh


def main():
    torch.manual_seed(3245)
    version = "initial"  
    device = "cuda:0"
    device = 'cpu'
    exp_folder = "02_23"
    epoch = 800
    batch_size = 512
    lr = 4e-4
    freq_dim = 15
    time_dim = 5    # 3, 5, or 7
    n_stride = 1    # 1 
    n_pool = 1
    n_channel = 128  # multiples of n_head (for multi-heads)
    n_cnn = 3       # 2,3, or 4
    n_fc = 3        # 2,3, or 4
    n_head = 4      # number of heads ( <= 4)
    #n_class = 4
        
    f_skip = False
    f_att = True
    f_MFCCDer = False
    f_max = False
    
    # set these values at features
    fs = 16000      # sampling freq
    n_fft = 512
    hop_length = 256
    speech_length = 5 
    
    n_frame = (speech_length * fs) // hop_length -1
    n_frame = (n_frame //2) *2

    # set model
    if f_att == False:
        net = Net3(freq_dim=freq_dim, time_dim=time_dim, n_frame=n_frame, n_stride=n_stride, n_pool=n_pool, n_channel=n_channel, \
               n_cnn=n_cnn, n_fc=n_fc, f_skip=f_skip, f_att=f_att, f_max=f_max).to(device)
    else:
        net = Net3_att(freq_dim=freq_dim, time_dim=time_dim, n_frame=n_frame, n_stride=n_stride, n_pool=n_pool, n_channel=n_channel, \
               n_cnn=n_cnn, n_fc=n_fc, f_skip=f_skip, f_att=f_att, f_max=f_max).to(device)
                   
    # Create a new workbook and select the active sheet
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet["A1"] = "Epoch"
    worksheet["B1"] = "Training Loss"
    worksheet["C1"] = "Validation Loss"
    worksheet["D1"] = "Training Acc"
    worksheet["E1"] = "Validation Acc"

    #net.load_state_dict(torch.load("/data3/ribka/SER/20/lrscheduler_1600.pt"))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)       # start with 0 and change
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True) 
    writer = SummaryWriter(exp_folder)

    dataset = CustomDataset("./train.csv", device=device, version=version)
    test_dataset = CustomDataset("./val.csv", device=device, version=version)
    
    normalise = True # feature normalisation
    n_mfcc = freq_dim
    if f_MFCCDer == False:
        glob = MFCC(n_mfcc, normalise=normalise, device=device)
    else:
        glob = MFCCDerivatives(n_mfcc, normalise=normalise, device=device)
    
    dataloader =  DataLoader(dataset, batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size)

    early_stopper = EarlyStopper()
    best_acc = float("-inf")
  

    for e in tqdm(range(epoch)):
        # reset for training
        count = 0
        total_loss = 0
        total_acc =0
        epoch_loss = 0
        epoch_acc= 0
        net.train()
        for wav, label in dataloader:
            feature = glob(wav)
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
        epoch_loss = total_loss/count  # average over epoch
        epoch_acc= total_acc/count  

        writer.add_scalar('Loss/train', epoch_loss, e)
        writer.add_scalar("Acc/train", epoch_acc, e)

        # reset for validation
        count = 0
        total_loss = 0
        total_acc = 0
        val_loss = 0
        val_acc= 0

        with torch.no_grad():
            net.eval()
            for wav, label in test_dataloader:
                feature = glob(wav)
                prediction = net(feature)
                loss = loss_fn(prediction, label)

                cur_batch_size = prediction.shape[0]
                count += cur_batch_size
                total_loss +=loss* cur_batch_size
                
                total_acc +=(prediction.argmax(dim=1)== label).type(torch.float).sum()

            # log every epoch
            val_loss = total_loss/count # average over epoch
            val_acc= total_acc/count

            writer.add_scalar('Loss/val', val_loss, e)
            writer.add_scalar("Acc/val", val_acc, e)

            if val_acc > best_acc:
                best_acc = val_acc  
                torch.save(net.state_dict(), exp_folder + "_best.pt") # save the best validation acc state dict
   

        if early_stopper.early_stop(val_loss):
            break
                  
        if(e % 2==0):
            print('  Tloss= ',str(epoch_loss),' Vacc= ',str(val_acc))
            
        if(e % 10==0):
            torch.save(net.state_dict(), exp_folder + "/" + str(e) + ".pt")

        
        scheduler.step(epoch_loss)
        
        # write to an Excel file

        worksheet["A"+str(e+2)] = e
        worksheet["B"+str(e+2)] = epoch_loss.item()
        worksheet["C"+str(e+2)] = val_loss.item()
        worksheet["D"+str(e+2)] = epoch_acc.item()
        worksheet["E"+str(e+2)] = val_acc.item()
        workbook.save(exp_folder + "-SA-30.xlsx")
    
if __name__ == "__main__":
    main()


