import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F


class MFCC(nn.Module):
    def __init__(self, params):
        super(MFCC, self).__init__()
        mel_args = params["mel_args"]
        n_mfcc = params["n_mfcc"]
        sample_rate = params["sample_rate"]
        log_mels = params["log_mels"]
        norm = params["norm"]
        self.feature_norm = params["feature_norm"]
     
        self.mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, log_mels=log_mels, melkwargs=mel_args, norm=norm)
        
    def forward(self, wav):
        feature = self.mfcc_transform(wav)

        if self.feature_norm:
            o_th = torch.exp(feature[...,0,:]/500.)-1
            am = torch.max(o_th) 
            o_th = torch.div(o_th, am + 0.01)
            above = feature[..., 1:,:]
            above_nor = 150         # normalize for non zeroth otders
            feature = torch.cat((o_th.unsqueeze(-2), above/above_nor), dim=-2)
        return feature


class Energy(nn.Module):
    def __init__(self, params):
        super(Energy, self).__init__()
        spec_args = params["spec_args"]

        self.spec_transform = torchaudio.transforms.Spectrogram(**spec_args)

    def forward(self, wav):
        spec = self.spec_transform(wav)
        energy = torch.sum(spec, dim=-2)
        return energy


class Mel(nn.Module):
    def __init__(self, params):
        super(Mel, self).__init__()
        mel_args = params["mel_args"]
        self.log = params["log"]
        self.log_offset = float(params["log_offset"])
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**mel_args)

    def forward(self, wav):
        mel_spec = self.mel_transform(wav)
        if self.log:
            mel_spec= torch.log(mel_spec + self.log_offset)

        return mel_spec


class Derivate(nn.Module):
    def __init__(self, params):
        super(Derivate, self).__init__()
        win_length = params["win_length"]
        mode= params["mode"]
        self.order = params["order"]
        self.Delta_transform = torchaudio.transforms.ComputeDeltas(win_length=win_length, mode=mode)
    
    def forward(self, feature):
        delta = self.Delta_transform(feature)
        feature = torch.cat((feature, delta), dim=-2)
        if not isinstance(self.order, int):
            delta_delta = self.Delta_transform(delta)
            feature = torch.cat((feature, delta_delta), dim=-2)
        return feature  


class Feature(nn.Module):
    def __init__(self, params):
        super(Feature, self).__init__()
        self.feature_type = params["feature_choice"]
        self.MFCC= MFCC(params["MFCC"]) if self.feature_type["MFCC"] else None
        self.Energy = Energy(params["Energy"]) if self.feature_type["Energy"] else None
        self.Mel = Mel(params["Mel"]) if self.feature_type["Mel"] else None
        self.Derivate = Derivate(params["Derivate"]) if self.feature_type["Derivate"] else None
        self.pitch_args = params["F0"]["pitch_args"]

    def forward(self, wav):
        mfcc = self.MFCC(wav) if self.MFCC else torch.FloatTensor([])
        energy = self.Energy(wav) if self.Energy else torch.FloatTensor([])
        mel = self.Mel(wav) if self.Mel else torch.FloatTensor([]) 
        pitch_nfcc = F.compute_kaldi_pitch(wav, **self.pitch_args) if self.feature_type["F0"] else None
        F0 = pitch_nfcc[..., 0] if pitch_nfcc is not None else torch.FloatTensor([])

        if F0.numel() > 0: 
            F0 = F0.unsqueeze(-2)
        if energy.numel() > 0: 
            energy = energy.unsqueeze(-2)
        
        feature = torch.cat([F0, mfcc, energy, mel], dim=-2)
        feature = self.Derivate(feature) if self.Derivate else feature
        return feature






