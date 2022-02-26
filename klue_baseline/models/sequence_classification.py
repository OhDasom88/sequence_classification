import argparse
import logging
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from overrides import overrides
from transformers import AutoModelForSequenceClassification, AutoModel

from klue_baseline.models import BaseTransformer, Mode


from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn.parameter import Parameter
import numpy as np

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

        
        # self.attention = BiAttention(self.arc_space, self.arc_space, 1)#512,512,1
        # self.bilinear = BiLinear(self.type_space, self.type_space, hparams.num_labels)
        self.attention = BiAttention(self.model.config.hidden_size, self.model.config.hidden_size, 1)#512,512,1
        self.bilinear = BiLinear(self.model.config.hidden_size, self.model.config.hidden_size, hparams.num_labels)


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
        
        # for i, sent in enumerate(inputs['input_ids']):
        #     for j, token in enumerate(sent):
        #         features
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


class BiAttention(nn.Module):
    def __init__(  # type: ignore[no-untyped-def]
        self, input_size_encoder: int, input_size_decoder: int, num_labels: int, biaffine: bool = True, **kwargs
    ) -> None:
        super(BiAttention, self).__init__()
        self.input_size_encoder = input_size_encoder
        self.input_size_decoder = input_size_decoder
        self.num_labels = num_labels
        self.biaffine = biaffine

        self.W_e = Parameter(torch.Tensor(self.num_labels, self.input_size_encoder))
        self.W_d = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder))
        self.b = Parameter(torch.Tensor(self.num_labels, 1, 1))
        if self.biaffine:
            self.U = Parameter(torch.Tensor(self.num_labels, self.input_size_decoder, self.input_size_encoder))
        else:
            self.register_parameter("U", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_e)
        nn.init.xavier_uniform_(self.W_d)
        nn.init.constant_(self.b, 0.0)
        if self.biaffine:
            nn.init.xavier_uniform_(self.U)

    def forward(
        self,
        input_d: torch.Tensor,
        input_e: torch.Tensor,
        mask_d: Optional[torch.Tensor] = None,
        mask_e: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert input_d.size(0) == input_e.size(0)
        batch, length_decoder, _ = input_d.size()
        _, length_encoder, _ = input_e.size()

        out_d = torch.matmul(self.W_d, input_d.transpose(1, 2)).unsqueeze(3)
        out_e = torch.matmul(self.W_e, input_e.transpose(1, 2)).unsqueeze(2)

        if self.biaffine:
            output = torch.matmul(input_d.unsqueeze(1), self.U)
            output = torch.matmul(output, input_e.unsqueeze(1).transpose(2, 3))
            output = output + out_d + out_e + self.b
        else:
            output = out_d + out_d + self.b

        if mask_d is not None:
            output = output * mask_d.unsqueeze(1).unsqueeze(3) * mask_e.unsqueeze(1).unsqueeze(2)

        return output


class BiLinear(nn.Module):
    def __init__(self, left_features: int, right_features: int, out_features: int):
        super(BiLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.W_l = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.W_r = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.bias = Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.constant_(self.bias, 0.0)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left: torch.Tensor, input_right: torch.Tensor) -> torch.Tensor:
        left_size = input_left.size()
        right_size = input_right.size()
        assert left_size[:-1] == right_size[:-1], "batch size of left and right inputs mis-match: (%s, %s)" % (
            left_size[:-1],
            right_size[:-1],
        )
        batch = int(np.prod(left_size[:-1]))

        input_left = input_left.contiguous().view(batch, self.left_features)
        input_right = input_right.contiguous().view(batch, self.right_features)

        output = F.bilinear(input_left, input_right, self.U, self.bias)
        output = output + F.linear(input_left, self.W_l, None) + F.linear(input_right, self.W_r, None)
        return output.view(left_size[:-1] + (self.out_features,))
