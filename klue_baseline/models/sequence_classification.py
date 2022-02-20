import argparse
import logging
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from overrides import overrides
from transformers import AutoModelForSequenceClassification, AutoModel

from klue_baseline.models import BaseTransformer, Mode

logger = logging.getLogger(__name__)


class SCTransformer(BaseTransformer):

    mode: str = Mode.SequenceClassification

    def __init__(self, hparams: Union[Dict[str, Any], argparse.Namespace], metrics: Dict[str, Any] = {}) -> None:
        if type(hparams) == dict:
            hparams = argparse.Namespace(**hparams)

        super().__init__(
            hparams,
            num_labels=hparams.num_labels,
            mode=self.mode,
            model_type=AutoModelForSequenceClassification,
            # model_type=AutoModel,# 20220215
            metrics=metrics,
        )

        # self.dense = nn.Linear(hparams.hidden_size, hparams.hidden_size)
        # self.dropout = nn.Dropout(hparams.hidden_dropout_prob)
        # self.out_proj = nn.Linear(hparams.hidden_size, hparams.num_labels)
        self.dense = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.model.config.hidden_size, hparams.num_labels)

        # self.dense1 = nn.Linear(768, 1)
        # self.dense2 = nn.Linear(128, hparams.num_labels)
        self.criterion = nn.CrossEntropyLoss()

    @overrides
    def forward(self, **inputs: torch.Tensor) -> Any:
        # return self.model(**inputs)
        # outputs = self.model(**inputs)# (tensor(1.0984), tensor([[ 0.0083,  0... 0.0256]]))
        # outputs[1].shape : torch.Size([64, 3])
        outputs = self.model.base_model(**{k:v for k, v in inputs.items() if k not in ['labels']})
        features = outputs[0]# torch.Size([64, 128, 768])
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        loss = self.criterion(x, inputs['labels'])
        return loss, x
 
    @overrides
    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}# (32,128), (32,128), (32)
        if self.is_use_token_type():
            inputs["token_type_ids"] = batch[2]#(32, 128)

        outputs = self(**inputs)
        loss = outputs[0]# torch.Size([]) 
        # outputs[1] torch.Size([32, 3])
        self.log("train/loss", loss)
        return {"loss": loss}

    @overrides
    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, data_type: str = "valid"
    ) -> Dict[str, torch.Tensor]:
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}# (64,128), (64,128), (64)
        if self.is_use_token_type():
            inputs["token_type_ids"] = batch[2]# #(64, 128)

        outputs = self(**inputs)
        loss, logits = outputs[:2]#(), (64, 3)

        self.log(f"{data_type}/loss", loss, on_step=False, on_epoch=True, logger=True)
        return {"logits": logits, "labels": inputs["labels"]}

    @overrides
    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]], data_type: str = "valid", write_predictions: bool = False
    ) -> None:
        labels = torch.cat([output["labels"] for output in outputs], dim=0)# output {'logits':(64, 3),'labels':(64)} len(outputs) 47
        preds = self._convert_outputs_to_preds(outputs)

        if write_predictions is True:
            self.predictions = preds

        self._set_metrics_device()
        for k, metric in self.metrics.items():
            metric(preds, labels)
            self.log(f"{data_type}/{k}", metric, on_step=False, on_epoch=True, logger=True)

    @overrides
    def _convert_outputs_to_preds(self, outputs: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        # logits: (B, num_labels)# len(outputs): 47, output["logits"].shape = (64, 3)
        logits = torch.cat([output["logits"] for output in outputs], dim=0)#(3000, 3) 64*46 + 56(last batch)
        return torch.argmax(logits, dim=1)

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
        BaseTransformer.add_specific_args(parser, root_dir)
        return parser
