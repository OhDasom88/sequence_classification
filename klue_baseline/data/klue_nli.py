import argparse
import json
import logging
import os
from typing import List, Optional

import torch
from overrides import overrides
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizer

from klue_baseline.data.base import DataProcessor, InputExample, InputFeatures, KlueDataModule
from klue_baseline.data.utils import convert_examples_to_features

import pandas as pd
import pathlib
from tqdm import tqdm
import re
from collections import defaultdict, Counter
from itertools import combinations_with_replacement
logger = logging.getLogger(__name__)


class KlueNLIProcessor(DataProcessor):

    origin_train_file_name: str = "klue-nli-v1.1_train.json"
    origin_dev_file_name: str = "klue-nli-v1.1_dev.json"
    origin_test_file_name: str = "klue-nli-v1.1_test.json"

    datamodule_type = KlueDataModule

    def __init__(self, args: argparse.Namespace, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(args, tokenizer)

    @overrides
    def get_train_dataset(self, data_dir: str, file_name: Optional[str] = None) -> TensorDataset:
        file_path = os.path.join(data_dir, file_name or self.origin_train_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "train")

    @overrides
    def get_dev_dataset(self, data_dir: str, file_name: Optional[str] = None) -> TensorDataset:
        file_path = os.path.join(data_dir, file_name or self.origin_dev_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "dev")

    @overrides
    def get_test_dataset(self, data_dir: str, file_name: Optional[str] = None) -> TensorDataset:
        file_path = os.path.join(data_dir, file_name or self.origin_test_file_name)

        
        if not os.path.exists(file_path):
            train_path = re.search(r'.+(?=sequence_classification)',__file__).group(0)+'sequence_classification'
            if not os.path.exists(train_path+'/data' + file_path): 
                logger.info("Test dataset doesn't exists. So loading dev dataset instead.")
                file_path = os.path.join(data_dir, self.hparams.dev_file_name or self.origin_dev_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "test")

    @overrides
    def get_labels(self) -> List[str]:
        return ["entailment", "contradiction", "neutral"]
        # return [''.join(sorted(i)) for i in combinations_with_replacement('nce', 5)]# 20220210

    def _create_dataset(self, file_path: str, dataset_type: str) -> TensorDataset:
        examples = self._create_examples(file_path, dataset_type)
        features = self._convert_features(examples)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)# torch.Size([24998, 128])
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)# torch.Size([24998, 128])
        # Some model does not make use of token type ids (e.g. RoBERTa)
        all_token_type_ids = torch.tensor(
            [0 if f.token_type_ids is None else f.token_type_ids for f in features], dtype=torch.long
        )#torch.Size([24998, 128])# t5-3b로 할때는 token_type를 반환하지 않음
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)# torch.Size([24998])

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)# len(dataset.tensors)4
        dataset.examples = examples# examples 속성 추가
        return dataset# [ i.shape for i in dataset.tensors] > [torch.Size([24998, 128]), torch.Size([24998, 128]), torch.Size([24998, 128]), torch.Size([24998])]

    def _create_examples(self, file_path: str, dataset_type: str) -> List[InputExample]:
        examples = []
        train_path = re.search(r'.+(?=sequence_classification)',__file__).group(0)+'sequence_classification'
        if '.csv' in file_path:
            if 'klue' in file_path:
                df = pd.read_csv(train_path+'/data'+ file_path)# special token을 반영
            else:
                if dataset_type == 'train':# 균등하게 sampling
                    df = df[:-3000]
                else:
                    df = df[-3000:]
        else:
            with open(f'{train_path}/data' + file_path, "r", encoding="utf-8") as f:
                data_lst = json.load(f)
                df = pd.DataFrame(data_lst)

        labels = defaultdict(int)# 새로운 레이블 정보
        # for data in tqdm(data_lst):
        for data in tqdm(df.itertuples()):
            data = data._asdict()
            guid =  data.get('guid')
            pre = data.get('_12' if self.hparams.file_name['DP'] else 'premise')
            hyp = data.get('_13' if self.hparams.file_name['DP'] else 'hypothesis')
            label = data.get('gold_label')
            # guid, pre, hyp, label = data[2], data[12], data[13], data.gold_label
            # guid, pre, hyp, label = data.index, data[6], data[7], data.label
            # guid, pre, hyp, label = data["guid"], data["premise"], data["hypothesis"], data["gold_label"]
            # guid, pre, hyp, label = data["guid"], data["premise"], data["hypothesis"], data['author'][0]+''.join(sorted([data[k][0] for k in ['label2','label3','label4','label5']]))
            # guid, pre, hyp, label = data["guid"], data["premise"], data["hypothesis"], ''.join(sorted([data[k][0] for k in ['author','label2','label3','label4','label5']]))# author를 따로 취급하기에는 일부 la
            labels[label]+= 1
            examples.append(InputExample(guid=guid, text_a=pre, text_b=hyp, label=label))
        # labels# 비대칭 분포 확인
        print(str(labels))# 비대칭 분포 확인
        
        # if dataset_type == 'train':# 학습데이터 셋에만 영어 데이터 추가
        #     # df = pd.read_csv(f'{str(pathlib.Path().resolve())}/data/MNLI/train.tsv', sep='\t', on_bad_lines='skip')
        #     # df1 = pd.read_csv(f'{str(pathlib.Path().resolve())}/data/MNLI/dev_matched.tsv', sep='\t', on_bad_lines='skip')
        #     # df2 = pd.read_csv(f'{str(pathlib.Path().resolve())}/data/MNLI/dev_mismatched.tsv', sep='\t', on_bad_lines='skip')
        #     # df = pd.concat([df1, df2])
        #     # df = pd.read_csv(f'{str(pathlib.Path().resolve())}/data/KorNLI/snli_1.0_train.ko.tsv', sep='\t', on_bad_lines='skip')
        #     df = pd.read_csv(f'{train_path}/data/KorNLI/multinli.train.ko.tsv', sep='\t', on_bad_lines='skip')
        #     df.dropna(inplace=True)
        #     for i, data in tqdm(enumerate(df.itertuples())):
        #         examples.append(InputExample(guid=i, text_a=data.sentence1, text_b=data.sentence2, label=data.gold_label))
        return examples

    def _convert_features(self, examples: List[InputExample]) -> List[InputFeatures]:
        return convert_examples_to_features(
            examples,
            self.tokenizer,
            label_list=self.get_labels(),
            max_length=self.hparams.max_seq_length,
            task_mode="classification",
        )

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
        parser = KlueDataModule.add_specific_args(parser, root_dir)
        parser.add_argument(
            "--max_seq_length",
            # default=512,
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        return parser
