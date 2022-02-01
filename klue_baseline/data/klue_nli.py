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
            logger.info("Test dataset doesn't exists. So loading dev dataset instead.")
            file_path = os.path.join(data_dir, self.hparams.dev_file_name or self.origin_dev_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "test")

    @overrides
    def get_labels(self) -> List[str]:
        return ["entailment", "contradiction", "neutral"]

    def _create_dataset(self, file_path: str, dataset_type: str) -> TensorDataset:
        examples = self._create_examples(file_path, dataset_type)
        features = self._convert_features(examples)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        # Some model does not make use of token type ids (e.g. RoBERTa)
        all_token_type_ids = torch.tensor(
            [0 if f.token_type_ids is None else f.token_type_ids for f in features], dtype=torch.long
        )
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        dataset.examples = examples
        return dataset

    def _create_examples(self, file_path: str, dataset_type: str) -> List[InputExample]:
        examples = []
        with open(f'{str(pathlib.Path().resolve())}' + '/data' + file_path, "r", encoding="utf-8") as f:
            data_lst = json.load(f)

        for data in tqdm(data_lst):
            guid, pre, hyp, label = data["guid"], data["premise"], data["hypothesis"], data["gold_label"]
            examples.append(InputExample(guid=guid, text_a=pre, text_b=hyp, label=label))
        
        
        # 모델은 그대로 영어 데이터셋만 추가해보기
        if dataset_type == 'train':# 학습데이터 셋에만 영어 데이터 추가
            # df = pd.read_csv(f'{str(pathlib.Path().resolve())}/data/MNLI/train.tsv', sep='\t', on_bad_lines='skip')
            df1 = pd.read_csv(f'{str(pathlib.Path().resolve())}/data/MNLI/dev_matched.tsv', sep='\t', on_bad_lines='skip')
            df2 = pd.read_csv(f'{str(pathlib.Path().resolve())}/data/MNLI/dev_mismatched.tsv', sep='\t', on_bad_lines='skip')
            df = pd.concat([df1, df2])
            df.dropna(inplace=True)
            for data in tqdm(df.itertuples()):
                examples.append(InputExample(guid=data.pairID, text_a=data.sentence1, text_b=data.sentence2, label=data.gold_label))
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
