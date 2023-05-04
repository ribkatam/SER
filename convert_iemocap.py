import pandas as pd
import re
import os 
path1 = "/data3/ribka/SER/IEMOCAP/Annotation"
path2 = "/data3/ribka/SER/IEMOCAP/Audio"

column_names = ["path", "emotion", "emotion_lb"]
data_df = pd.DataFrame(columns=column_names)
count = 0
big_folder = os.listdir(path1)
big_folder.sort()
c_sur= c_sad=c_fru=c_neu = c_ang= c_hap=c_dis=c_exc=c_fea=c_xxx = c_oth=0

for session in big_folder:
    inside_session = os.listdir(os.path.join(path1, session))
    inside_session.sort()

    for ann_path in inside_session:

        with open(os.path.join(path1, session, ann_path)) as f:
            content = f.read()
        info_line = re.compile(r'\[.+\]')
        relevant_lines = re.findall(info_line, content)
        for  line in relevant_lines[1:]:
            time, audio_path, category, dimensions = line.strip().split("\t")
            start_time, end_time = time[1:-1].split("-")
            val, act, dom = dimensions[1:-1].split(",")
            if category == "neu":
                emotion = 0
                c_neu +=1
            elif category =="fru":
                emotion = 1
                c_fru+=1
            elif category == "ang":
                emotion = 3
                c_ang+=1
            elif category == "sad":
                emotion = 2
                c_sad+=1
            elif category == "hap":
                emotion = 1
                c_hap+=1
            elif category == "exc":
                emotion = 1
                category = "hap"
                c_exc+=1
            elif category == "sur":
                emotion = 6
                c_sur+=1
            elif category == "fea":
                emotion = 7
                c_fea+=1
            elif category == "dis":
                emotion = 8
                c_dis+=1
            elif category == "xxx":
                emotion = 9
                c_xxx+=1
            elif category == "oth":
                emotion = 9
                c_oth+=1
            else:
                emotion = 10

            if category not in ["neu", "fru", "ang", "sad", "hap", "exc", "sur", "fea", "xxx"]:
                print(category)
            data_df.loc[count] = [os.path.join(path2, session, ann_path.split(".")[0], audio_path +".wav"), emotion, category]
            count+=1
            
print("cneu%d, cfru%d, cang%d, csad%d, chap%d, cexc%d, csur%d, cfea%d, cdis%d, cxxx%d, coth%d", c_neu, c_fru, c_ang, c_sad,c_hap, c_exc, c_sur, c_fea, c_dis, c_xxx, c_oth) 
 
data_df.to_csv("/data3/ribka/SER/IEMOCAP/session(2,3,4,5).csv", index=False)