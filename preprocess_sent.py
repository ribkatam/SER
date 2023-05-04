import torch 
from pydub import AudioSegment, silence
import pandas as pd
import torchaudio
import ffmpeg
import os
from tqdm import tqdm
import pickle

 
val_annotation_folder_path = "/data1/ribka/AFF_WILD2/VA_validation_annotation"
annotation_saving_path = "/data1/ribka/AFF_WILD/Sentence/val_annotation.pkl"
video_folder_path1 = "/data1/ribka/AFF_WILD2/batch1_vid/"
video_folder_path2 = "/data1/ribka/AFF_WILD2/batch2_vid/"
audio_saving_path = "/data1/ribka/AFF_WILD/Sentence/ValAudio"


file_list = os.listdir(val_annotation_folder_path)
file_list.sort()
all_frames_df = pd.DataFrame() 
hop_length = 10

for file_name in tqdm(file_list):
    annotation_file_path = os.path.join(val_annotation_folder_path, file_name) 
    file_name = file_name.split(".")[0]
    annotation_df = pd.read_csv(annotation_file_path)
    num_vid_frames = len(annotation_df)
    total_duration = 0

    try: 
        video_file_path = os.path.join(video_folder_path1, file_name + ".mp4")
        audio_clip = AudioSegment.from_file(video_file_path)
    except FileNotFoundError:
        try:
            video_file_path = os.path.join(video_folder_path2, file_name +".mp4")
            audio_clip = AudioSegment.from_file(video_file_path)
        except FileNotFoundError:
            try:
                video_file_path = os.path.join(video_folder_path1, file_name +".avi")
                audio_clip = AudioSegment.from_file(video_file_path)
            except FileNotFoundError:
                video_file_path = os.path.join(video_folder_path1, file_name +".avi")
                audio_clip = AudioSegment.from_file(video_file_path)

    
    probe = ffmpeg.probe(video_file_path)
    frame_rate = int((probe["streams"][0]["avg_frame_rate"]).split("/")[0])
    vid_frame_duration = 1/frame_rate * 1000
    
   
    for i, x in enumerate(silence.split_on_silence(audio_clip, keep_silence=True,min_silence_len=100)):
        duration = len(x)
        wav_name = os.path.join(audio_saving_path, file_name+"_{}".format(i)+".wav")
        x.export(wav_name, parameters = ["-ac", "1", "-ar", "16000"], format="wav")
        wav, sr = torchaudio.load(wav_name)
        stft =torch.stft(wav.view(-1), n_fft=400, hop_length=160, center=True, onesided=True)
        num_frames = stft.shape[1]
        #print(stft.shape)
        frame_num_list = [j for j in range(num_frames)]
        label_list = []

        for k in range(num_frames):
            annotation_index = (total_duration + (k * hop_length))//vid_frame_duration
            if(annotation_index > num_vid_frames -1):
                annotation_index =  num_vid_frames -1
            valence = annotation_df.loc[0, "valence"]
            arousal = annotation_df.loc[0, "arousal"]
            label_list.append([valence, arousal])
            #print("index is {}".format(annotation_index))
            
            # if (annotation_index > 100):
            #     break  
        index = [[wav_name]*num_frames]
        #print(len(index[0]))
        index.append(frame_num_list)
        
        #print(len(label_list))
        
        df = pd.DataFrame(label_list, index=index, columns=["valence", "arousal"])
        all_frames_df = pd.concat([all_frames_df, df])
        total_duration  +=duration
        #print(total_duration)
 

all_frames_df.to_csv(annotation_saving_path.split(".")[0]+"csv")
output = open(annotation_saving_path, 'wb')
pickle.dump(all_frames_df, output)
output.close()
