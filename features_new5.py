import torch
import torch.nn as nn
import numpy as np
import torchaudio
from  torch.nn.utils.rnn import pad_sequence 
from scipy.fftpack import dct, dst


class MFCC(nn.Module):
    def __init__(self, n_mfcc, normalise=False, device="cpu"):
        super(MFCC, self).__init__()
        self.mel_args = {"n_fft": 512, "hop_length": 256, "n_mels": 80, "center": False, "power": 2}  # n_fft = 512
        self.n_mfcc = n_mfcc
        self.normalise = normalise
        self.device=device
        self.mfcc_transform = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc= self.n_mfcc, log_mels=True, melkwargs=self.mel_args, norm=None).to(self.device)
        
    def forward(self, wav):
        feature = self.mfcc_transform(wav)
        if self.normalise:
            # mfcc_mean = torch.mean(feature, dim=-1, keepdim=True)# calculate max over time dim for each feature and mfcc order separately.
            # mfcc_std = torch.std(feature, dim=-1, keepdim=True)
            # feature = (feature - mfcc_mean) / mfcc_std


            # mfcc_max, _ = torch.max(feature, dim=-1, keepdim=True)  
            # mfcc_min, _ = torch.min(feature, dim=-1, keepdim=True)
            # feature = (feature - mfcc_min) / (mfcc_max - mfcc_min)
            
            #o_min = 930
            #o_max = 1530
            #o_th = (o_th - o_min) / (o_max - o_min)
            #o_th = torch.where(o_th < 0,0,o_th)
            
            o_th = torch.exp(feature[:,0,:]/500.)-1
            am, tt = torch.max(o_th,1) 
            o_th = torch.div(torch.transpose(o_th,0,1),am+0.01)
            o_th = torch.transpose(o_th,0,1)

            above = feature[:,1:,:]
            above_nor = 150         # normalize for non zeroth otders
            feature = torch.cat((o_th.unsqueeze(1), above/above_nor), dim=1)
        
        return feature


class Mel(nn.Module):
    def __init__(self, n_mel, normalise=False, device="cpu"):
        super(Mel, self).__init__()
        self.mel_args = {"n_fft": 512, "hop_length": 256, "n_mels": n_mel, "center": False, "power": 1}   # n_fft = 512
        self.normalise = normalise
        self.device=device
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, **self.mel_args).to(self.device)

    def forward(self, wav):
        feature = self.mel_transform(wav)
        if self.normalise:
            mel_mean = torch.mean(feature, dim=-1, keepdim=True)# calculate mean over time dim for each sample and mel band separately.
            mel_std = torch.std(feature, dim=-1, keepdim=True)
            feature = (feature - mel_mean) / mel_std

            # mel_max, _ = torch.max(feature, dim=-1, keepdim=True)  
            # mel_min, _ = torch.min(feature, dim=-1, keepdim=True)
            # feature = (feature - mel_min) / (mel_max - mel_min)
        
        return feature
    

class MFCCDerivatives(MFCC):
    def __init__(self, n_mfcc, normalise=False, device="cpu"):
        super(MFCCDerivatives, self).__init__(n_mfcc=n_mfcc, normalise=False, device=device)
        self.normalise = normalise
        self.delta_transform = torchaudio.transforms.ComputeDeltas(win_length =3, mode="constant")

    def forward(self, wav):
        mfcc = super(MFCCDerivatives, self).forward(wav)
        delta = self.delta_transform(mfcc)
        delta_delta = self.delta_transform(delta)
        feature = torch.cat((mfcc, delta, delta_delta), dim=1)
        
        if self.normalise:
            # feature_mean = torch.mean(feature, dim=-1, keepdim=True)
            # feature_std = torch.std(feature, dim=-1, keepdim=True)
            # feature = (feature - feature_mean) / feature_std

            # o_th = feature [:,0,:]
            # above = feature[:,1:,:]
            # max_0th, _= torch.max(o_th, dim=-1, keepdim=True)
            # max_above, _ = torch.max(above.reshape(feature.shape[0],-1), dim=-1, keepdim=True)
            # feature = torch.cat(((o_th/max_0th).unsqueeze(1), above/max_above.unsqueeze(1)), dim=1)
            
            o_th = torch.exp(feature[:,0,:]/500.)-1
            am, tt = torch.max(o_th,1) 
            o_th = torch.div(torch.transpose(o_th,0,1),am+0.01)
            o_th = torch.transpose(o_th,0,1)
            above = feature[:,1:,:]
            above_nor = 150         # normalize for non zeroth otders
            feature = torch.cat((o_th.unsqueeze(1), above/above_nor), dim=1)

        return feature


class GlobFeature(nn.Module):
    def __init__(self, n_mfcc=5, k=10, device="cpu", normalise=False):
        super(GlobFeature, self).__init__()
        self.mfcc_transform = MFCC(n_mfcc=n_mfcc, normalise=normalise, device=device)
        self.device = device
        self.k = k
        self.normalise = normalise

    def forward(self, wav):
        feature = self.mfcc_transform(wav)
        dct_feature = dct(feature.cpu().numpy(), axis =-1)[:, :, :self.k]
        dst_feature = dst(feature.cpu().numpy(), axis =-1)[:, :, :self.k]
        concat_feature = np.concatenate([dct_feature, dst_feature], axis=-1)
        flattened_feature = torch.flatten(torch.from_numpy(concat_feature), start_dim=1)

        return flattened_feature.to(self.device)
   


class FreqMeanSpread(nn.Module):
    def __init__(self, normalise=False, device="cpu"):
        super(FreqMeanSpread, self).__init__()
        self.mel_args = {"n_fft": 512, "hop_length": 256, "n_mels": 80, "center": False, "power": 1}
        self.spec_transform = torchaudio.transforms.MelSpectrogram(**self.mel_args)
        self.normalise = normalise
        self.device=device

    def forward(self, wavs):
        spec = self.spec_transform(wavs)
        mean = torch.mean(spec, dim=1)
        std = torch.std(spec, dim=1)
        concated = torch.stack((mean, std), dim=1).to(self.device)

        if self.normalise:
            ms_mean = torch.load("ms_mean.pt").to(self.device)
            ms_std = torch.load("ms_std.pt").to(self.device)
            concated = (concated - ms_mean) / ms_std

        return concated


class GlobFeatureWithMS(nn.Module):
    def __init__(self, n_mfcc=5, k=10, device="cpu", normalise=False):
        super(GlobFeature, self).__init__()
        self.mfcc_transform = MFCC(n_mfcc=n_mfcc, normalise=normalise)
        self.mean_std = FreqMeanSpread(normalise=normalise)
        self.device = device
        self.k = k
        self.normalise = normalise

    def forward(self, wav):
        mfcc = self.mfcc_transform(wav)
        mean_std = self.mean_std(wav)

        feature = torch.cat((mfcc, mean_std), dim=1)
        dct_feature = dct(feature.numpy(), axis =-1)[:, :, :self.k]
        dst_feature = dst(feature.numpy(), axis =-1)[:, :, :self.k]
        concat_feature = np.concatenate([dct_feature, dst_feature], axis=-1)
        flattened_feature = torch.flatten(torch.from_numpy(concat_feature), start_dim=1)
        
        return flattened_feature.to(self.device)



    