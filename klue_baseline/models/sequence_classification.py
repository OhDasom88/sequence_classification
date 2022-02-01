import argparse
import logging
from typing import Any, Dict, List, Union

import torch
from overrides import overrides
from transformers import AutoModelForSequenceClassification

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
            metrics=metrics,
        )

    @overrides
    def forward(self, **inputs: torch.Tensor) -> Any:
        return self.model(**inputs)

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
