import torchaudio
import torch
import torch.nn as nn

import pandas as pd
import os
import yaml
from tqdm import tqdm
from feature import Feature

def main():
    global Feature_class 
    global version
    global time
    global min_len
    global max_len
    global saving_folder


    with open("param.yaml", "r") as stream:
            params = yaml.safe_load(stream)

    version_params = params["audio"]["version"]
    sr = params["audio"]["sr"]
    min_len = params["audio"]["min_len"] * sr
    max_len = params["audio"]["max_len"] * sr
    
    audio_path =  params["saving"]["audio_path"]
    old_train_ann = params["saving"]["old_train_ann"]
    old_val_ann = params["saving"]["old_val_ann"]
    saving_folder = params["saving"]["folder"]
    new_train_ann = params["saving"]["new_train_ann"]
    new_val_ann = params["saving"]["new_val_ann"]
    feature_params = params["feature"]
  
   
 
    Feature_class = Feature(feature_params)

    for x in version_params:
        if version_params[x]:
            version = x
            time = int(version_params[x]) * sr
    

    save_feature(audio_path, old_train_ann, new_train_ann)
    save_feature(audio_path, old_val_ann, new_val_ann)


def save_feature(audio_path, ann_path, new_path):
    data = pd.read_csv(ann_path)
    column_names = ["path", "emotion", "emotion_lb"]
    new_df = pd.DataFrame(columns=column_names)
    for idx in tqdm(range(len(data))):
        wav_path = data.iloc[idx][0]
        absolute_path = wav_path.replace("IEMOCAP", audio_path)
        wav, sr = torchaudio.load(absolute_path, normalize=False)
        wav = wav[0].float()
        length = wav.shape[0]
        
        if version == "initial":
            if length < min_len:
               continue
            if length < time:
                pad_dim = time - length 
                wav  = nn.functional.pad(wav, (0, pad_dim), mode="constant", value=0)
            wav = wav[:time]
            
            assert len(wav) == time
            
        elif version == "pad":
            if length < min_len:
               continue

            if length < max_len:   
                pad_dim = max_len - length 
                wav  = nn.functional.pad(wav, (0, pad_dim), mode="constant", value=0)

            assert len(wav) ==max_len
        

        elif version == "middle":
            if length < min_len:
               continue

            if length < time:
                pad_dim = time - length
                pad_left = pad_dim//2
                pad_right = pad_dim-pad_left
                wav  = nn.functional.pad(wav, (pad_left, pad_right), mode="constant", value=0)
            else: 
                pad_dim = length - time
                pad_left = pad_dim//2
                pad_right = pad_left + time
                wav = wav[pad_left:pad_right]

            assert len(wav) == time
        
            
        else:  
            if length < min_len: 
                continue
         
        
        feature = Feature_class(wav)
        
        
        emotion = data.iloc[idx][1] 
        emotion_lb = data.iloc[idx][2]
        path = wav_path.replace("IEMOCAP", saving_folder)
    
        splitted = path.split("/")
        joined = os.path.join(*splitted[:-1])
    
        os.makedirs(joined, exist_ok=True)
        saving_path = os.path.join(joined, splitted[-1].split(".")[0] + ".pt" )
        
        torch.save(feature, saving_path)
        new_df.loc[idx] = [saving_path, emotion, emotion_lb] 
        
    new_df.to_csv(new_path, index=False)

    
if __name__ == "__main__":
    main()