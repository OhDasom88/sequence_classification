from re import A
from numpy import append
import pandas as pd
import json
from tqdm import tqdm

with open('/home/dasomoh88/sequence_classification/data/output.json') as f:
    data = json.load(f)

sent_id = -1
examples = []
for name in ('test', 'train'):
    newCols = []
    df = pd.read_csv(f'/home/dasomoh88/sequence_classification/data/dacon/{name}_data.csv')
    for i, row in tqdm(enumerate(df.itertuples())):
        newRow = []
        newCols.append(newRow)
        for col in [2,3]:#premise, hypothesis
            sent_id += 1
            sent = []
            word_id = 0
            words = row[col].split()
            chnk_info = data[sent_id][-1]
            while word_id < len(words):
                chnk = []
                chnk_idx = 0
                while len(chnk_info)>chnk_idx:
                    if word_id in chnk_info[chnk_idx]:
                        chk = sorted(chnk_info.pop(chnk_idx))# 이미 사용된 chnk 정보는 지우기
                        for chk_id in chk:
                            chnk.append(words[chk_id])
                        break
                    chnk_idx += 1
                if len(chnk)>0:
                    sent.append(' '.join(chnk))
                    word_id+=len(chnk)
                else:
                    sent.append(words[word_id])
                    word_id+=1
            newRow.append(' [word] '.join(sent))
    pd.concat([df, pd.DataFrame(newCols)], axis=1).to_csv(f'/home/dasomoh88/sequence_classification/data/dacon/{name}_data_new.csv')

