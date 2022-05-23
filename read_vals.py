from asyncore import read
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt



def most_frequent(List):
    return max(set(List), key = List.count)

em_map={}
em_others_map={}
FPS=24
NO_EMOTION = "None"

files = ["blerllgmgq.xls","hbccpwuhmw.xls","hsxgcrfnwl.xls",
"krrojpfhdt.xls","mcddkuubod.xls","uejrkohryz.xls","vppyctfohr.xls","wzsgmztdvr.xls"]
for f in files:
    data = pd.read_excel(f)
    df_with_nan=pd.DataFrame(data)
    df=df_with_nan.replace(-1,np.nan)
    for i in range(len(df.columns)):
        col_temp = df.iloc[:,i]
        #print(col)
        col = col_temp.dropna(axis=0)
        for j in range(0,len(col)-FPS):
            em1,em2=col.iloc[j],col.iloc[j+FPS]
            if (f'{em1}_{em2}') in em_map:
                em_map[(f'{em1}_{em2}')]+=1
            else:
                em_map[(f'{em1}_{em2}')]=1
            
        for j in range(0,len(col_temp)-1):
            em1 = col_temp.iloc[j+1]
            # if nan then ignore
            if(str(em1) == "nan"):
                continue
            # get all other columns except the current one
            row = df.iloc[j]
            np_row = row.to_numpy()
            other_items = np.delete(np_row,i)
            filtered = list(filter(lambda v: v==v, other_items))
            other_emotion = NO_EMOTION
            if(len(filtered) > 0):
                other_emotion = most_frequent(filtered)
        
            if(f'{em1}_{other_emotion}' in em_others_map):
                em_others_map[f'{em1}_{other_emotion}'] += 1
            else:
                em_others_map[f'{em1}_{other_emotion}'] = 1

total = sum(list(em_map.values()))
emission_total = sum(list(em_others_map.values()))

prob_map={}
emission_map={}
for em in em_map:
    prob_map[em]=round(em_map[em]/total,5)

for em in em_others_map:
    emission_map[em]=round(em_others_map[em]/total,5)

names = list(prob_map.keys())
values = list(prob_map.values())

plt.bar(range(len(prob_map)), values, tick_label=names)
plt.xticks(rotation = 90,fontsize= 5) 
plt.show()
















