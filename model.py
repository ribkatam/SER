# Last updated on 03/30/2023

import torch
import torch.nn as nn
import torch.nn.functional as F
from runner_mockingjay import get_mockingjay_model



class ConvAttBlock(nn.Module):
    def __init__(self, params):
        super(ConvAttBlock, self).__init__()
        conv_params = params["ConvBlock"]
        n_in_channels = conv_params["n_in_channels"]
        n_out_channels = conv_params["n_out_channels"]
        kernel_size_1 = eval(conv_params["kernel_size_1"])
        kernel_size_other = eval(conv_params["kernel_size_other"])
        stride= conv_params["stride"]
        padding = conv_params["padding"]
        n_layers = conv_params["num_layers"]
        bias = conv_params["bias"]
        pool_size = eval(conv_params["pool_size"])

        fc_params = params["FC"]
        num_fc_layers = fc_params["num_layers"]

        att_params = params["Attention"]
        num_att_layers = att_params["num_layers"]
        n_td = att_params["n_td"]
        n_bu = att_params["n_bu"]
        n_mid = att_params["n_mid"]
        self.n_att = att_params["n_att"]
        self.alpha = att_params["alpha"]

        self.conv_list = nn.ModuleList([nn.Conv2d(in_channels=n_in_channels, out_channels=n_out_channels, kernel_size=kernel_size_1, stride=stride, padding=(0, kernel_size_1[1]//2), bias=bias)])                               
        self.conv_list = self.conv_list.extend([nn.Conv2d(in_channels=n_out_channels, out_channels=n_out_channels, kernel_size=kernel_size_other, stride=stride, padding=padding, bias=bias)
                                                   for _ in range (n_layers - 1)])
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size)


        for i in range(num_fc_layers + 1):
            fc_params["f"+str(i)] = fc_params["f"+str(i)]

        self.fc = nn.ModuleList([nn.Linear(fc_params["f"+ str(x)] , fc_params["f"+ str(x+1)])
                                for x in range (num_fc_layers)])
    

        self.att_bu = nn.Linear(n_out_channels, n_bu)
        self.att_td = nn.Linear(4, n_td)
        self.att_tdbu = nn.ModuleList([nn.Linear(n_bu + n_td , n_mid)])
        self.att_tdbu = self.att_tdbu.extend([nn.Linear(n_mid, n_mid) for _ in range (num_att_layers -2)])   
        self.att_tdbu = self.att_tdbu.append(nn.Linear(n_mid, n_out_channels))               


    def forward(self, feature):
        device = feature.get_device()
        x = feature.unsqueeze(1) # N,H,S --> N,C,H,S
       
        for module in self.conv_list:
            x = F.relu(self.max_pool(module(x)))
           
        
        #x = torch.squeeze(x, -2)   # N,C,1,S --> N,C,S
        x = torch.flatten(x, start_dim=-2)   # N,C,1,S --> N,C,S
        x = torch.transpose(x,1,-1)  # # N,S,C <-- N,C,S
     
        x_bu = self.att_bu(x)
     
        p_labels = torch.tensor([1490, 1561, 1151, 1027]) / (1490+1561+1151+1027)   # class probability
        p_labels = p_labels.repeat(x.size()[0]*x.size()[1],1)
        p_labels = torch.reshape(p_labels,(x.size()[0],x.size()[1],4)).to(device)

        for _ in range(self.n_att):
            attended = torch.cat((x_bu, self.att_td(p_labels)), dim=-1)

            for module in self.att_tdbu:
                attended = F.tanh(module(attended))
           
            att_residual = x * (self.alpha * attended + 1.)
            y = att_residual
            for i, module in enumerate(self.fc):
                if i != len(self.fc)-1:
                    y = F.relu(module(y))
                else:
                    y = module(y)

            attended = y                                                                                
            p_labels = F.softmax(attended, dim=1)

        attended = torch.transpose(attended,-1,-2) # N,S,C --> N,C,S
        glob_pool = nn.AvgPool1d(attended.shape[-1])
        prediction = glob_pool(attended) 
        prediction = torch.squeeze(prediction, -1)  #N,C,1 --> N, C
        
        return prediction


class BERTConvAtt(nn.Module):
    def __init__(self, params):
        super(BERTConvAtt, self).__init__()
        self.convblock = ConvAttBlock(params)
        path = params["BERT"]["path"]
        self.mj = get_mockingjay_model(from_path=path) 
        
        

    def forward(self, melspec):
       
        x = torch.transpose(melspec, -1, -2) # N,C,S --> N,S,C
        feature = self.mj.forward(x, all_layers=False, tile=True) 
        feature= torch.transpose(feature, -1, -2)  # N,C,S <-- N,S,C
        prediction = self.convblock(feature)
        return prediction
        