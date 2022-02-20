""" Usage
$ python inference.py --data_dir /data \
                      --model_dir /model \
                      --output_dir /output \
                      [args ...]
"""
import argparse
import os
import tarfile

import torch
from dataloader import KlueDpDataLoader
from model import AutoModelforKlueDp
from transformers import AutoConfig, AutoTokenizer
from utils import flatten_prediction_and_labels, get_dp_labels, get_pos_labels

from tqdm import tqdm
from collections import defaultdict
import json

# KLUE_DP_OUTPUT = "output.csv"  # the name of output file should be output.csv
KLUE_DP_OUTPUT = "dp_output_klue_train.json"  # the name of output file should be output.csv


def load_model(model_dir, args):
    # extract tar.gz
    # model_name = args.model_tar_file
    # tarpath = os.path.join(model_dir, model_name)
    # tar = tarfile.open(tarpath, "r:gz")
    # tar.extractall(path=model_dir)

    config = AutoConfig.from_pretrained(os.path.join(model_dir, "config.json"))
    model = AutoModelforKlueDp(config, args)
    model.load_state_dict(torch.load(os.path.join(model_dir, "dp-model.bin"), map_location='cpu'))
    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    # device setup
    num_gpus = torch.cuda.device_count()
    use_cuda = num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")

    # load model
    model = load_model(model_dir, args)
    model.to(device)
    model.eval()

    # load KLUE-DP-test
    kwargs = {"num_workers": num_gpus, "pin_memory": True} if use_cuda else {}
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    klue_dp_dataset = KlueDpDataLoader(args, tokenizer, data_dir)
    klue_dp_test_loader = klue_dp_dataset.get_test_dataloader(
        args.test_filename, **kwargs
    )

    
    dp_labels = get_dp_labels()
    # inference
    head_pred = []
    type_pred = []
    chunk_pred = []
    for i, batch in tqdm(enumerate(klue_dp_test_loader)):
        input_ids, masks, ids, max_word_length = batch
        input_ids = input_ids.to(device)
        attention_mask, bpe_head_mask, bpe_tail_mask, mask_e, mask_d = (
            mask.to(device) for mask in masks
        )
        head_ids, type_ids, pos_ids = (id.to(device) for id in ids)

        batch_size, _ = head_ids.size()
        batch_index = torch.arange(0, batch_size).long()

        out_arc, out_type = model(
            bpe_head_mask,
            bpe_tail_mask,
            pos_ids,
            head_ids,
            max_word_length,
            mask_e,
            mask_d,
            batch_index,
            input_ids,
            attention_mask,
        )

        heads = torch.argmax(out_arc, dim=2).tolist()
        types = torch.argmax(out_type, dim=2).tolist()

        for j in range(batch_size):
            head_ = [h for h in heads[j] if h!=0]+[0]# h = 0 >> PAD or root
            tmp = None
            tmpSet = set()# chunk 후보 피지배소 목록
            chunk_custom = set() # chunk 목록
            for k, h in enumerate(heads[j]):# 각 사례별 참조관계 확인
                if tmp is None: 
                    tmp =h
                    tmpSet.add(k)#지배소 초기화 및 피지배소 목록 갱신
                elif tmp == h: 
                    tmpSet.add(k)# 이전 어절과 동일한 지배소를 가지는 경우 피지배소 목록 갱신
                elif tmp != h:# 동일한 지배소를 가지는 피지배소 목록에서 구문정보와 기능정보 제약을 점검
                    if len(tmpSet) <2: 
                        tmpSet.clear()
                        tmp=h
                        tmpSet.add(k)
                        continue# 뭉칠 피지배소 목록의 크기는 최소 2 이상이어야 함
                    
                    phr, func = defaultdict(set), set()
                    for l, idx in enumerate(tmpSet):
                        dp_label = dp_labels[types[j][idx]].split('_')
                        phr[dp_label[0]].add(idx)
                        if len(dp_label) ==2 and dp_label[-1].strip() in ['CMP', 'MOD', 'AJT']:
                            func.add(idx)
 
                    # 구문 정보가 같거나 기능정보가 모두 보어 (CMP), 체언수식어(MOD), 용언수식어(AJT)
                    for _ , v in phr.items():# 구문 정보가 같거나 
                        if len(v)>1:
                            tmp_sorted = sorted(v)
                            chunk_based_on_phr = set()
                            for l in range(len(tmp_sorted)-1):
                                if (tmp_sorted[l+1] - tmp_sorted[l])==1:# 구문정보가 같은 연속적인 어절
                                    chunk_based_on_phr.add(tmp_sorted[l])
                                    chunk_based_on_phr.add(tmp_sorted[l+1])
                                elif len(chunk_based_on_phr) > 0:
                                    chunk_custom.add(tuple(chunk_based_on_phr))# 구문정보가 같은 연속적인 어절을 chunk 후보 목록에 추가
                                    chunk_based_on_phr.clear()

                    if len(chunk_based_on_phr)>0:
                        chunk_custom.add(tuple(chunk_based_on_phr))
                        chunk_based_on_phr.clear()

                    if len(func) <2:
                        tmpSet.clear()
                        tmp=h
                        tmpSet.add(k)
                        continue

                    tmp_sorted = sorted(func)#  보어(CMP), 체언수식어(MOD), 용언수식어(AJT)에 속하는 어절 정보의 연속성 평가
                    chunk_based_on_func = set()
                    for l in range(len(tmp_sorted)-1):#기능정보가 모두
                        if (tmp_sorted[l+1] - tmp_sorted[l])==1:# 구문정보가 같은 연속적인 어절
                            chunk_based_on_func.add(tmp_sorted[l])
                            chunk_based_on_func.add(tmp_sorted[l+1])
                        elif len(chunk_based_on_func) > 0:# 불연속 점 발견 >> 지금까지 모인chunk 후보군을 구문기준 chunk 후보군에 취합
                            for chk_ in chunk_custom:
                                if set(chk_).issubset(chunk_based_on_func):# 구문 또는 기능 정보 기준 chunk의 크기가 큰것 기준으로 chunk 수정
                                    chunk_custom.remove(chk_)
                                    chunk_custom.add(tuple(chunk_based_on_func))# 구문정보가 같은 연속적인 어절을 chunk 후보 목록에 추가
                                    chunk_based_on_func.clear()
                                    break
                                elif set(chk_).issuperset(chunk_based_on_func):# chunk 크기가 큰것 기준
                                    chunk_based_on_func.clear()
                                    break
                            if len(chunk_based_on_func)>0:# chunk 후보군에서 포함관계가 없었던 경우, 교집합 부분 점검
                                chunk_based_on_func_tmp = set()
                                for chk_ in chunk_custom:
                                    if len(set(chk_).intersection(chunk_based_on_func))>1:# 교집합
                                        chunk_based_on_func_tmp.update(chunk_based_on_func)
                                        chunk_based_on_func.clear()# 목적 chunk 초기화
                                        break
                                if len(chunk_based_on_func)==0:# 목적 chunk와 교집합을 가지는 집합이 있는 경우
                                    chunk_based_on_func_tmp_len = 0 
                                    while chunk_based_on_func_tmp_len != len(chunk_based_on_func_tmp):
                                        chunk_based_on_func_tmp_len = len(chunk_based_on_func_tmp)
                                        del_chk = []
                                        for chk_ in chunk_custom:
                                            if len(set(chk_).intersection(chunk_based_on_func_tmp)) > 1:
                                                chunk_based_on_func_tmp.update(chk_)# 교집합이 있는 청크를 하나로 모음
                                                del_chk.append(chk_)# 합쳐진 chunk는 삭제할 목록에 추가
                                        for del_c in del_chk:
                                            chunk_custom.remove(del_c)# 합쳐진 chunk는 삭제
                                    chunk_custom.add(tuple(chunk_based_on_func_tmp))# 합집합을 chunk목록에 추가
                                else:
                                    chunk_custom.add(tuple(chunk_based_on_func))# 기존 목록에서 전혀 교차하는 부분이 없는 경우 그대로 추가
                    
                    if len(chunk_based_on_func) > 0:# 아직 chunk로 이동이 덜된 요소가 남아 있는경우
                        for chk_ in chunk_custom:
                            if set(chk_).issubset(chunk_based_on_func):# 구문 또는 기능 정보 기준 chunk의 크기가 큰것 기준으로 chunk 수정
                                chunk_custom.remove(chk_)
                                chunk_custom.add(tuple(chunk_based_on_func))# 구문정보가 같은 연속적인 어절을 chunk 후보 목록에 추가
                                chunk_based_on_func.clear()
                                break
                            elif set(chk_).issuperset(chunk_based_on_func):# chunk 크기가 큰것 기준
                                chunk_based_on_func.clear()
                                break
                        if len(chunk_based_on_func)>0:# chunk 후보군에서 포함관계가 없었던 경우, 교집합 부분 점검
                            chunk_based_on_func_tmp = set()
                            for chk_ in chunk_custom:
                                if len(set(chk_).intersection(chunk_based_on_func))>1:# 교집합
                                    chunk_based_on_func_tmp.update(chunk_based_on_func)
                                    chunk_based_on_func.clear()# 목적 chunk 초기화
                                    break
                            if len(chunk_based_on_func)==0:# 목적 chunk와 교집합을 가지는 집합이 있는 경우
                                chunk_based_on_func_tmp_len = 0 
                                while chunk_based_on_func_tmp_len != len(chunk_based_on_func_tmp):
                                    chunk_based_on_func_tmp_len = len(chunk_based_on_func_tmp)
                                    del_chk = []
                                    for chk_ in chunk_custom:
                                        if len(set(chk_).intersection(chunk_based_on_func_tmp)) > 1:
                                            chunk_based_on_func_tmp.update(chk_)# 교집합이 있는 청크를 하나로 모음
                                            del_chk.append(chk_)# 합쳐진 chunk는 삭제할 목록에 추가
                                    for del_c in del_chk:
                                        chunk_custom.remove(del_c)# 합쳐진 chunk는 삭제
                                chunk_custom.add(tuple(chunk_based_on_func_tmp))# 합집합을 chunk목록에 추가
                            else:
                                chunk_custom.add(tuple(chunk_based_on_func))# 기존 목록에서 전혀 교차하는 부분이 없는 경우 그대로 추가

                    tmpSet.clear()# 피지배소 목록 초기화
                    tmp=h# 지배소 갱신
                    tmpSet.add(k)
                if h ==0:
                    break# root를 참조하는 어절 : 문단의 마지막

            head_pred.append(head_)
            type_pred.append(types[j][:len(head_)])
            chunk_pred.append(tuple(chunk_custom))
    
    # write results to output_dir
    with open(os.path.join(output_dir, KLUE_DP_OUTPUT), "w", encoding="utf8") as f:
        json.dump([([(ht[0], dp_labels[ht[-1]]) for ht in zip(h,t)],c) for h,t,c in zip(head_pred, type_pred, chunk_pred)], f)
        # json.dump([(h, [dp_labels[ht[-1]] for ht in t],c) for h,t,c in zip(head_pred, type_pred, chunk_pred)], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Container environment
    parser.add_argument(
        # "--data_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/data")
        "--data_dir", type=str, default='/home/dasomoh88/sequence_classification/data/klue_dp/klue-dp-v1.1'
    )
    parser.add_argument(
        # "--model_dir", type=str, default="./model"
        "--model_dir", type=str, default="/home/dasomoh88/sequence_classification/model/dp_model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        # default=os.environ.get("SM_OUTPUT_DATA_DIR", "/output"),
        default='/home/dasomoh88/sequence_classification/inference/output/',
    )

    # inference arguments
    parser.add_argument(
        "--model_tar_file",
        type=str,
        default="klue_dp_model.tar.gz",
        help="it needs to include all things for loading baseline model & tokenizer, \
             only supporting transformers.AutoModelForSequenceClassification as a model \
             transformers.XLMRobertaTokenizer or transformers.BertTokenizer as a tokenizer",
    )
    parser.add_argument(
        "--test_filename",
        # default="klue-dp-v1.1_test.tsv",
        default="klue-dp-v1.1_dev.tsv",
        # default="klue-dp-v1.1_dev_sample_10.tsv",
        type=str,
        help="Name of the test file (default: klue-dp-v1.1_test.tsv)",
    )
    parser.add_argument(
        "--eval_batch_size"
        , default=64
        , type=int
    )

    # model-specific arguments
    parser = AutoModelforKlueDp.add_arguments(parser)

    # parse args
    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    inference(data_dir, model_dir, output_dir, args)
