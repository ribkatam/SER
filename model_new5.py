# Last updated on 02/23/2023

import torch.nn as nn
import torch

class Net3(nn.Module):
    def __init__(self, freq_dim, time_dim, n_frame, n_stride, n_pool, n_channel, n_cnn, n_fc, f_skip, f_att, f_max):
        super(Net3, self).__init__()
        n_out = 128
        n_cnn = 3
        n_fc = 3
        m_out = n_out
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=n_channel, kernel_size=(freq_dim,time_dim), stride=(1,1), \
                              padding=(0,time_dim//2), bias = False)  
        self.cnn2 = nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=(1,time_dim), stride=(1,1), \
                              padding=(0,time_dim//2), bias = False)  
        self.cnn3 = nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=(1,time_dim), stride=(1,1), \
                              padding=(0,time_dim//2), bias = False)  
        self.cnn4 = nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=(1,time_dim), stride=(1,1), \
                              padding=(0,time_dim//2), bias = False)  
        self.cnn5 = nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=(1,time_dim), stride=(1,1), \
                              padding=(0,time_dim//2), bias = False)  
        self.max_pool = nn.MaxPool2d(kernel_size=(1,2))     # done after 2nd and 4th CNNs
        self.glob_pool1 = nn.AvgPool2d(kernel_size = (1,n_frame//2))  
        self.glob_pool2 = nn.AvgPool2d(kernel_size = (1,n_frame//4))
        self.glob_pool3 = nn.AvgPool2d(kernel_size = (1,n_frame//8))
        self.glob_pool4 = nn.AvgPool2d(kernel_size = (1,n_frame//16))
        self.glob_pool5 = nn.AvgPool2d(kernel_size = (1,n_frame//32))
        self.flatten = nn.Flatten() 
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)
        if n_fc == 2:
            self.fc=nn.Sequential(nn.Linear(m_out, 32), nn.ReLU(), nn.Linear(32, 4))
        elif n_fc ==3:
            self.fc=nn.Sequential(nn.Linear(m_out, 64), nn.ReLU(), nn.Linear(64,16), nn.ReLU(), nn.Linear(16, 4))
        elif n_fc ==4:
            self.fc=nn.Sequential(nn.Linear(m_out, 128), nn.ReLU(), nn.Linear(128,24), nn.ReLU(), nn.Linear(24,16), \
                                  nn.ReLU(), nn.Linear(16,4))

    def forward(self, feature):
        feature = feature.unsqueeze(1)  # N, H, W  --> N, C, H, W
        x = self.max_pool(self.relu(self.cnn1(feature)))
        x = self.max_pool(self.relu(self.cnn2(x)))
        
        #if self.f_skip == True:
        #    skip2 = self.glob_pool2(x)
        n_cnn = 3
        if n_cnn > 2:
            x = self.max_pool(self.relu(self.cnn3(x)))
            #skip3 = self.glob_pool3(x)
            if n_cnn > 3:
                x = self.max_pool(self.relu(self.cnn4(x)))
                #skip4 = self.glob_pool4(x)
                if n_cnn > 4:
                    x = self.max_pool(self.relu(self.cnn5(x)))
                    #skip5 = self.glob_pool5(x)

        #concatenated = self.flatten(torch.cat((skip1, skip2), dim=1))
        #prediction = self.fc(concatenated)
        
        prediction = self.fc(torch.transpose(x,1,3))
        prediction = self.glob_pool3(torch.transpose(prediction,1,3))
        prediction = torch.squeeze(prediction)

        return prediction


###################################################################################
class Net3_att(nn.Module):
    def __init__(self, freq_dim, time_dim, n_frame, n_stride, n_pool, n_channel, n_cnn, n_fc, f_skip, f_att, f_max):
        super(Net3_att, self).__init__()
        n_out = 128
        n_cnn = 3
        n_fc = 3
        m_out = n_out
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=n_channel, kernel_size=(freq_dim,time_dim), stride=(1,1), \
                              padding=(0,time_dim//2), bias = False)  
        self.cnn2 = nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=(1,time_dim), stride=(1,1), \
                              padding=(0,time_dim//2), bias = False)  
        self.cnn3 = nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=(1,time_dim), stride=(1,1), \
                              padding=(0,time_dim//2), bias = False)  
        self.cnn4 = nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=(1,time_dim), stride=(1,1), \
                              padding=(0,time_dim//2), bias = False)  
        self.cnn5 = nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=(1,time_dim), stride=(1,1), \
                              padding=(0,time_dim//2), bias = False)  
        self.max_pool = nn.MaxPool2d(kernel_size=(1,2))     # done after 2nd and 4th CNNs
        self.glob_pool1 = nn.AvgPool2d(kernel_size = (1,n_frame//2))  
        self.glob_pool2 = nn.AvgPool2d(kernel_size = (1,n_frame//4))
        self.glob_pool3 = nn.AvgPool2d(kernel_size = (1,n_frame//8))
        self.glob_pool4 = nn.AvgPool2d(kernel_size = (1,n_frame//16))
        self.glob_pool5 = nn.AvgPool2d(kernel_size = (1,n_frame//32))
        self.flatten = nn.Flatten() 
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)
        if n_fc == 2:
            self.fc=nn.Sequential(nn.Linear(m_out, 32), nn.ReLU(), nn.Linear(32, 4))
        elif n_fc ==3:
            self.fc=nn.Sequential(nn.Linear(m_out, 64), nn.ReLU(), nn.Linear(64,16), nn.ReLU(), nn.Linear(16, 4))
        elif n_fc ==4:
            self.fc=nn.Sequential(nn.Linear(m_out, 128), nn.ReLU(), nn.Linear(128,24), nn.ReLU(), nn.Linear(24,16), \
                                  nn.ReLU(), nn.Linear(16,4))
        self.att_bu = nn.Linear(m_out,n_out)
        self.att_td = nn.Linear(4,n_out)
        self.att_tdbu = nn.Sequential(nn.Linear(n_out+n_out,16), nn.Tanh(), nn.Linear(16,16), \
                                      nn.Tanh(), nn.Linear(16,m_out))


    def forward(self, feature):
        n_cnn = 3
        alpha = 0.3
        n_att = 3
        feature = feature.unsqueeze(1)  # N, H, W  --> N, C, H, W
        x = self.max_pool(self.relu(self.cnn1(feature)))
        x = self.max_pool(self.relu(self.cnn2(x)))
        
        #if self.f_skip == True:
        #    skip2 = self.glob_pool2(x)
        
        if n_cnn > 2:
            x = self.max_pool(self.relu(self.cnn3(x)))
            #skip3 = self.glob_pool3(x)
            if n_cnn > 3:
                x = self.max_pool(self.relu(self.cnn4(x)))
                #skip4 = self.glob_pool4(x)
                if n_cnn > 4:
                    x = self.max_pool(self.relu(self.cnn5(x)))
                    #skip5 = self.glob_pool5(x)
        
        ###
        x = torch.squeeze(torch.transpose(x,1,3))
        p_labels = torch.tensor([1490, 1561, 1151, 1027]) / (1490+1561+1151+1027)   # class probability
        p_labels = p_labels.repeat(x.size()[0]*x.size()[1],1)
        p_labels = torch.reshape(p_labels,(x.size()[0],x.size()[1],4))
         
        for aa in range(n_att):
            attended = torch.cat((self.att_bu(x), self.att_td(p_labels)), dim=2)
            attended = self.fc( x * (alpha * self.att_tdbu(attended) +1.))
            #attended = self.dropout(attended)
            p_labels = self.softmax(attended)
        
        # repeat 
              
        prediction = self.glob_pool3(torch.transpose(attended,1,2))
        prediction = torch.squeeze(prediction)

        return prediction

###################################################################################
#   add multi-head for top-down attention
class Net3_att_mh(nn.Module):
    def __init__(self, freq_dim, time_dim, n_frame, n_stride, n_pool, n_channel, n_cnn, n_fc, f_skip, f_att, f_max):
        super(Net3_att_mh, self).__init__()
        n_out = 128
        n_cnn = 3
        n_fc = 3
        m_out = n_out
        n_head = 4      # 2 or 4
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=n_channel, kernel_size=(freq_dim,time_dim), stride=(1,1), \
                              padding=(0,time_dim//2), bias = False)  
        self.cnn2 = nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=(1,time_dim), stride=(1,1), \
                              padding=(0,time_dim//2), bias = False)  
        self.cnn3 = nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=(1,time_dim), stride=(1,1), \
                              padding=(0,time_dim//2), bias = False)  
        self.cnn4 = nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=(1,time_dim), stride=(1,1), \
                              padding=(0,time_dim//2), bias = False)  
        self.cnn5 = nn.Conv2d(in_channels=n_channel, out_channels=n_channel, kernel_size=(1,time_dim), stride=(1,1), \
                              padding=(0,time_dim//2), bias = False)  
        self.max_pool = nn.MaxPool2d(kernel_size=(1,2))     # done after 2nd and 4th CNNs
        self.glob_pool1 = nn.AvgPool2d(kernel_size = (1,n_frame//2))  
        self.glob_pool2 = nn.AvgPool2d(kernel_size = (1,n_frame//4))
        self.glob_pool3 = nn.AvgPool2d(kernel_size = (1,n_frame//8))
        self.glob_pool4 = nn.AvgPool2d(kernel_size = (1,n_frame//16))
        self.glob_pool5 = nn.AvgPool2d(kernel_size = (1,n_frame//32))
        self.flatten = nn.Flatten() 
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)
        if n_fc == 2:
            self.fc=nn.Sequential(nn.Linear(m_out, 32), nn.ReLU(), nn.Linear(32, 4))
        elif n_fc ==3:
            self.fc=nn.Sequential(nn.Linear(m_out, 64), nn.ReLU(), nn.Linear(64,16), nn.ReLU(), nn.Linear(16, 4))
        elif n_fc ==4:
            self.fc=nn.Sequential(nn.Linear(m_out, 128), nn.ReLU(), nn.Linear(128,24), nn.ReLU(), nn.Linear(24,16), \
                                  nn.ReLU(), nn.Linear(16,4))
        self.att_bu = nn.Linear(m_out,n_out)
        self.att_td = nn.Linear(4,n_out)
        self.att_tdbu1 = nn.Sequential(nn.Linear((n_out+n_out)//n_head,16), nn.Tanh(), nn.Linear(16,16), \
                                      nn.Tanh(), nn.Linear(16,m_out//n_head))
        self.att_tdbu2 = nn.Sequential(nn.Linear((n_out+n_out)//n_head,16), nn.Tanh(), nn.Linear(16,16), \
                                      nn.Tanh(), nn.Linear(16,m_out//n_head))
        self.att_tdbu3 = nn.Sequential(nn.Linear((n_out+n_out)//n_head,16), nn.Tanh(), nn.Linear(16,16), \
                                      nn.Tanh(), nn.Linear(16,m_out//n_head))
        self.att_tdbu4 = nn.Sequential(nn.Linear((n_out+n_out)//n_head,16), nn.Tanh(), nn.Linear(16,16), \
                                      nn.Tanh(), nn.Linear(16,m_out//n_head))

    def forward(self, feature):
        n_cnn = 3
        alpha = 0.
        n_att = 3
        n_head = 4      # 2 or 4
        n_out = 128
        m_out = 128
        m_oh = (n_out+n_out)//n_head
        m_oh2 = m_oh //2
        feature = feature.unsqueeze(1)  # N, H, W  --> N, C, H, W
        x = self.max_pool(self.relu(self.cnn1(feature)))
        x = self.max_pool(self.relu(self.cnn2(x)))
        
        #if self.f_skip == True:
        #    skip2 = self.glob_pool2(x)
        
        if n_cnn > 2:
            x = self.max_pool(self.relu(self.cnn3(x)))
            #skip3 = self.glob_pool3(x)
            if n_cnn > 3:
                x = self.max_pool(self.relu(self.cnn4(x)))
                #skip4 = self.glob_pool4(x)
                if n_cnn > 4:
                    x = self.max_pool(self.relu(self.cnn5(x)))
                    #skip5 = self.glob_pool5(x)

        #concatenated = self.flatten(torch.cat((skip1, skip2), dim=1))
        #prediction = self.fc(concatenated)
        
        ###
        x = torch.squeeze(torch.transpose(x,1,3))
        p_labels = torch.tensor([1490, 1561, 1151, 1027]) / (1490+1561+1151+1027)   # class probability
        p_labels = p_labels.repeat(x.size()[0]*x.size()[1],1)
        p_labels = torch.reshape(p_labels,(x.size()[0],x.size()[1],4))

        # multi-head
        for na in range(n_att):
            attcon = torch.cat((self.att_bu(x), self.att_td(p_labels)), dim=2)

            attended = torch.zeros(attcon.size()[0],attcon.size()[1],n_out)
            #print(attended.shape)
            
            for nh in range(n_head):
                attend = attcon[:,:,nh*m_oh:(nh+1)*m_oh]
                if nh == 0:
                    attended[:,:,nh*m_oh2:(nh+1)*m_oh2] = self.att_tdbu1(attend)
                elif nh == 1:
                    attended[:,:,nh*m_oh2:(nh+1)*m_oh2] = self.att_tdbu2(attend)
                elif nh == 2:
                    attended[:,:,nh*m_oh2:(nh+1)*m_oh2] = self.att_tdbu3(attend)
                elif nh == 3:
                    attended[:,:,nh*m_oh2:(nh+1)*m_oh2] = self.att_tdbu4(attend)
            xx = self.fc(x * (alpha * attended +1.))
            p_labels = self.softmax(xx)
        
        prediction = self.glob_pool3(torch.transpose(xx,1,2))
        prediction = torch.squeeze(prediction)

        return prediction
