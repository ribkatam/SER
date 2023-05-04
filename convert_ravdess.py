import os
import pandas as pd

def convertRAVDESS(speech_path, annotation_file_path):
    column_names = ["path", "emotion", "emotion_lb", "emotion_intensity"]

    data_df = pd.DataFrame(columns=column_names)
    dir_list_speech = os.listdir(speech_path)
    dir_list_speech.sort()
    count = 0

    for actor_folder in dir_list_speech:
        file_list = os.listdir(speech_path + actor_folder)

        for f in file_list:

            labels = f.split('.')[0].split('-')
            path = speech_path + actor_folder + "/" + f
            emotion = int(labels[2])
            emotion_intensity = int(labels[3])

            if emotion in (1, 2) :
                emotion = 0
                emotion_lb = "neutral"
            else:
                if emotion == 3:
                    emotion_lb = "happy"
                elif emotion == 4 :
                    emotion_lb = "sad"
                elif emotion == 5:
                    emotion_lb = "angry"
                elif emotion == 6:
                    emotion_lb = "fearful" 
                elif emotion == 7:
                    emotion_lb = "digust"
                elif emotion ==8:
                    emotion_lb = "surprised"
                emotion -=2
      

            values = [path, emotion, emotion_lb, emotion_intensity]
            data_df.loc[count]=  values
            count +=1


    data_df.to_csv(annotation_file_path, index=False)



train_speech_path= "/data3/ribka/SER/RAVDESS/Training/Audio_Speech_Actors_01-24/"  
train_annotation_file_path = "/data3/ribka/SER/train_ann.csv"


val_speech_path= "/data3/ribka/SER/RAVDESS/Validation/Audio_Speech_Actors_01-24/"
val_annotation_file_path = "/data3/ribka/SER/val_ann.csv"

test_speech_path= "/data3/ribka/SER/RAVDESS/Testing/Audio_Speech_Actors_01-24/"
test_annotation_file_path = "/data3/ribka/SER/test_ann.csv"

convertRAVDESS(train_speech_path, train_annotation_file_path)
convertRAVDESS(val_speech_path,val_annotation_file_path)
convertRAVDESS(test_speech_path, test_annotation_file_path)

