import pandas as pd
import torch
import os 
from tqdm import tqdm

def main():
    audio_path = "/data3/ribka/SER/IEMOCAP"
    data = pd.read_csv("val2.csv")
    column_names = ["path", "emotion", "emotion_lb"]
    saving_folder = "IEMOCAP_MOCAPNAN"
    new_ann_path = "val_face_nan.csv"
    data_df = pd.DataFrame(columns=column_names)

    for idx in tqdm(range (len(data))):
        path = data.iloc[idx][0].replace("/Audio","").replace("wav","txt").replace("IEMOCAP", audio_path)
        path = path.split("/")
        path.insert(6, "MOCAP_rotated")
        txt_path = os.path.join(*path)
        path = path[5:]
        path.insert(0, saving_folder)
        # print(path)
        joined = os.path.join(*path[:-1])
        os.makedirs(joined, exist_ok=True)

   
        saving_path = os.path.join(joined, path[-1].split(".")[0] + ".pt" )
  
        change_to_tensor(txt_path, saving_path)
        data_df.loc[idx] = [saving_path, data.iloc[idx][1], data.iloc[idx][2]]

    data_df.to_csv(new_ann_path, index=False)




def change_to_tensor(txt_path, saving_path):
    feature_list=[]
    with open("/" + txt_path) as f:
        content = f.read()
        split = content.split("\n")
        for x in range(2, len(split)-1):
            row_list = []
            row = split[x].split()
            for y in range(2, len(row)):
                if row[y] == "NaN":
                   value = float("nan")  
                else:
                    value = eval(row[y]) 
                # if value == float("nan"):
                #     print(value)
                row_list.append(value)
    
            feature_list.append(row_list)

        torch.save(torch.Tensor(feature_list), saving_path)



    
if __name__ == "__main__":
    main()