import pandas as pd
import json
from tqdm import tqdm
import re

defaultDir = re.search(r'.+(?=sequence_classification)',__file__).group(0)+'sequence_classification'

# sent_id = -1
# examples = []
for name in ('test', 'train'):# KLUE
# for name in ('test',):# DACON
    sent_id = -1
    newCols = []

    with open(f'{defaultDir}/data/dacon/klue-nli-v1_1_{name}_json.json') as f:
    # with open(f'{defaultDir}/data/dacon/{name}_data_csv.json') as f:
        data = json.load(f)

    # # /home/dasomoh88/sequence_classification/data/dacon/klue-nli-v1.1_test.json
    with open(f'{defaultDir}/data/dacon/klue-nli-v1_1_{name}.json') as f:
        dataOrigin = json.load(f)

    # dataOrigin = pd.read_csv(f'{defaultDir}/data/dacon/{name}_data.csv')

    # Dacon
    # for i, row in tqdm(enumerate(dataOrigin.itertuples())):
    # KLUE
    for i, row in tqdm(enumerate(dataOrigin)):
        newRow = []
        newCols.append(newRow)
        newRow.append(data[sent_id][-1])# tag representaion
        # for col in [2,3]:#csv DACON <<  premise, hypothesis
        for col in ['premise','hypothesis']:# json KLUE << premise, hypothesis
            sent_id += 1
            sent = []
            word_id = 0
            words = row[col].split()
            chnk_info = data[sent_id][-2]
            
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

    # pd.concat([df, pd.DataFrame(newCols)], axis=1).to_csv(f'{defaultDir}/data/dacon/{name}_from_klue_new_with_dp.csv')
    # json from klue
    pd.concat([pd.DataFrame(dataOrigin), pd.DataFrame(newCols)], axis=1).to_csv(f'{defaultDir}/data/dacon/{name}_from_klue_new_with_dp_v2.csv')
    # csv from dacon
    # pd.concat([dataOrigin, pd.DataFrame(newCols)], axis=1).to_csv(f'{defaultDir}/data/dacon/{name}_from_dacon_with_dp_v2.csv')
