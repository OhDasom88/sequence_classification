import argparse
import logging
import os
import sys
from pathlib import Path
from tkinter import W
from typing import List, Optional, Dict, Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from klue_baseline import KLUE_TASKS
from klue_baseline.utils import Command, LoggingCallback
import wandb
from datetime import datetime as dt
import torch


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def add_general_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
    parser.add_argument(
        "--task",
        type=str,
        default='klue-nli',# 20220131
        # required=True,# 20220131
        help=f"Run one of the task in {list(KLUE_TASKS.keys())}",
    )
    parser.add_argument(
        "--output_dir",
        # default=None,
        default='output',
        type=str,
        # required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--gpus",
        default = [0] if torch.cuda.is_available() else None,
        nargs="+",
        type=int,
        help="Select specific GPU allocated for this, it is by default [] meaning none",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision instead of 32-bit",
    )
    parser.add_argument(
        "--num_sanity_val_steps",
        type=int,
        default=2,
        help="Sanity check validation steps (default 2 steps)",
    )
    parser.add_argument("--n_tpu_cores", dest="tpu_cores", type=int)
    parser.add_argument("--max_grad_norm", dest="gradient_clip_val", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    # parser.add_argument("--metric_key", type=str, default="loss", help="The name of monitoring metric")
    parser.add_argument("--metric_key", type=str, default="accuracy", help="The name of monitoring metric")
    parser.add_argument(
        "--patience",
        # default=5,
        default=10,# 20220221
        type=int,
        help="The number of validation epochs with no improvement after which training will be stopped.",
    )
    parser.add_argument(
        "--early_stopping_mode",
        choices=["min", "max"],
        default="max",
        type=str,
        help="In min mode, training will stop when the quantity monitored has stopped decreasing; in max mode it will stop when the quantity monitored has stopped increasing;",
    )
    return parser


def make_klue_trainer(
    args: argparse.Namespace,
    extra_callbacks: List = [],
    checkpoint_callback: Optional[pl.Callback] = None,
    logging_callback: Optional[pl.Callback] = None,
    **extra_train_kwargs,
) -> pl.Trainer:
    pl.seed_everything(args.seed)

    # Logging
    # csv_logger = CSVLogger(args.output_dir, name=args.task)
    # args.output_dir = csv_logger.log_dir
    wandb_logger = WandbLogger(save_dir=args.output_dir, name=args.task)

    if logging_callback is None:
        logging_callback = LoggingCallback()

    # add custom checkpoints
    metric_key = f"valid/{args.metric_key}"
    if checkpoint_callback is None:
        filename_for_metric = "{" + metric_key + ":.2f}"

        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(args.output_dir).joinpath("checkpoint"),
            monitor=metric_key,
            filename="{epoch:02d}-{step}=" + filename_for_metric,
            save_top_k=2,
            # mode="max",
            mode="max" if metric_key == 'accuracy' else 'min',
        )
    # Andrew Ng Proj >> Early Stopping
    # what early stopping does is by stopping halfway you have only a mid-size rate w. 
    # And so similar to L2 regularization by picking a neural network with smaller norm for your parameters w
    # , hopefully your neural network is over fitting less.
    early_stopping_callback = EarlyStopping(monitor=metric_key, patience=args.patience, mode=args.early_stopping_mode)
    extra_callbacks.append(early_stopping_callback)

    # train_params: Dict[str, Any] = {}
    # sweep config item 값을 hyper Parameter로 전달하기 위해 수정
    train_params: Dict[str, Any] = pl.Trainer.default_attributes()
    for k, v in train_params.items():
        if args.get(k):
            train_params[k] = args.get(k) 

    if args.fp16:
        train_params["precision"] = 16

    # Set GPU & Data Parallel
    args.num_gpus = 0 if args.gpus is None else len(args.gpus)
    # if args.num_gpus > 1:
    #     train_params["accelerator"] = "dp"
    # Sweep 적용중
    if args.num_gpus > 1:
        train_params["accelerator"] = "dp"
    train_params["val_check_interval"] = 0.25  # check validation set 4 times during a training epoch
    # train_params["num_sanity_val_steps"] = args.num_sanity_val_steps
    # train_params["accumulate_grad_batches"] = args.accumulate_grad_batches
    train_params["profiler"] = extra_train_kwargs.get("profiler", None)
    train_params["weights_summary"] = None
    train_params["callbacks"] = [logging_callback] + extra_callbacks
    train_params["logger"] = wandb_logger
    train_params["checkpoint_callback"] = checkpoint_callback
    train_params["progress_bar_refresh_rate"] = 20
    # train_params["progress_bar_refresh_rate"] = 1

    # return pl.Trainer.from_argparse_args(
    #     args,
    #     weights_summary=None,
    #     callbacks=[logging_callback] + extra_callbacks,
    #     # logger=csv_logger,
    #     logger=wandb_logger,
    #     checkpoint_callback=checkpoint_callback,
    #     **train_params,
    # )
    return pl.Trainer(**train_params,)


def log_args(args: argparse.Namespace) -> None:
    args_dict = vars(args)
    max_len = max([len(k) for k in args_dict.keys()])
    fmt_string = "\t%" + str(max_len) + "s : %s"
    logger.info("Arguments:")
    for key, value in args_dict.items():
        logger.info(fmt_string, key, value)

def my_train_func():
    wandb.init()
    args = wandb.config
    task = KLUE_TASKS.get(wandb.config.get('task'), None)
    command = wandb.config.get('command')


    # log_args(args)
    trainer = make_klue_trainer(args)
    task.setup(args, command)
    
    # wandb.watch(task.model, criterion=None, log="all", log_freq=1000, idx=None, log_graph=(False))    
    wandb.watch(task.model, criterion=None, log="all", log_freq=500, idx=None, log_graph=(False))    

    if command == Command.Train:
        logger.info("Start to run the full optimization routine.")
        trainer.fit(**task.to_dict())

        # load the best checkpoint automatically
        trainer.get_model().eval_dataset_type = "valid"
        val_results = trainer.test(test_dataloaders=task.val_loader, verbose=False)[0]
        print("-" * 80)

        output_val_results_file = os.path.join(args.output_dir, "val_results.txt")
        with open(output_val_results_file, "w") as writer:
            for k, v in val_results.items():
                writer.write(f"{k} = {v}\n")
                print(f" - {k} : {v}")
        print("-" * 80)

    elif command == Command.Evaluate:
        trainer.test(task.model, test_dataloaders=task.val_loader)
    elif command == Command.Test:
        trainer.test(task.model, test_dataloaders=task.test_loader)

def main() -> None:
    # command = sys.argv[1].lower()
    command = 'train'
    # command = 'test'

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        # "command",
        "--command",#20220131
        default=command,#20220131
        type=str,
        help=f"Whether to run klue with command ({Command.tolist()})",
    )
    if command in ["--help", "-h"]:
        parser.parse_known_args()
    elif command not in Command.tolist():
        raise ValueError(f"command is positional argument. command list: {Command.tolist()}")

    # Parser (general -> data -> model)
    parser = add_general_args(parser, os.getcwd())
    parsed, _ = parser.parse_known_args()
    task_name = parsed.task

    task = KLUE_TASKS.get(task_name, None)

    if not task:
        raise ValueError(f"task_name is positional argument. task list: {list(KLUE_TASKS.keys())}")

    parser = task.processor_type.add_specific_args(parser, os.getcwd())
    parser = task.model_type.add_specific_args(parser, os.getcwd())
    args = parser.parse_args()


    sweep_configuration = {
        "name": f"{args.model_name_or_path}_{dt.now().strftime('%m%d_%H:%M')}",
        # "metric": {"name": "valid/accuracy", "goal": "maximize"},
        "metric": {"name": f"valid/{args.metric_key}", "goal": "minimize" if args.metric_key == 'loss' else 'maximize'},# args.metric_key
        # "method": "grid",
        "method": "bayes",
        "parameters": {},
        # 'early_terminate':{'type': 'hyperband' , 'max_iter': 27, 's': 2}
        # "early_terminate": {"type": "hyperband", "min_iter": 3,},
    }
    sweep_configuration['parameters'] = {k:{'values': [v]} for k , v in vars(args).items() if k not in ['encoder_layerdrop', 'decoder_layerdrop', 'dropout','attention_dropout']}

    if vars(args).get('model_name_or_path') is None:
        # sweep_configuration['parameters'].update({'model_name_or_path':{'distribution': 'categorical', 'values':['kykim/electra-kor-base','klue/roberta-small', 'klue/roberta-base','klue/roberta-large']}})
        sweep_configuration['parameters'].update({'model_name_or_path':{'distribution': 'categorical', 'values':['klue/roberta-small']}})
        sweep_configuration.update({"name": f"{sweep_configuration['parameters']['model_name_or_path']}_{dt.now().strftime('%m%d_%H:%M')}",})

    else:
        sweep_configuration['parameters'].update({'model_name_or_path':{'distribution': 'categorical', 'values':[vars(args).get('model_name_or_path')]}})

    # sweep_configuration['parameters'].update({'train_file_name':{'distribution': 'categorical', 'values':['train_data_new.csv']}})# ['word'] token 
    # sweep_configuration['parameters'].update({'train_file_name':{'distribution': 'categorical', 'values':['train_from_klue_new_with_dp.csv']}})# ['word'] token 
    # sweep_configuration['parameters'].update({'dev_file_name':{'distribution': 'categorical', 'values':['test_from_klue_new_with_dp.csv']}})# partial train data(2000) ['word'] token  
    # sweep_configuration['parameters'].update({'test_file_name':{'distribution': 'categorical', 'values':['test_from_klue_new_with_dp.csv']}})#

    sweep_configuration['parameters'].update(
        {
            'file_name':{
                'distribution': 'categorical'
                , 'values':[
                    {'DP':True,'filenames':('train_from_klue_new_with_dp.csv','test_from_klue_new_with_dp.csv','test_from_klue_new_with_dp.csv')},
                    # {'DP':False,'filenames':('klue-nli-v1_1_train.json','klue-nli-v1_1_test.json','klue-nli-v1_1_test.json')},
                ]
            }
        }
    )#
    # sweep_configuration['parameters'].update({'train_file_name':{'distribution': 'categorical', 'values':['klue-nli-v1.1_train.json']}})#
    # sweep_configuration['parameters'].update({'dev_file_name':{'distribution': 'categorical', 'values':['klue-nli-v1.1_dev.json']}})#
    # sweep_configuration['parameters'].update({'test_file_name':{'distribution': 'categorical', 'values':['klue-nli-v1.1_test.json']}})#
    sweep_configuration['parameters'].update({'fp16':{'distribution': 'categorical', 'values':[torch.cuda.is_available()]}})#, False# GPU가 사용가능할때만
    # sweep_configuration['parameters'].update({'adafactor':{'distribution': 'categorical', 'values':[vars(args).get('adafactor')]}})# True, False
    # sweep_configuration['parameters'].update({'adam_epsilon':{'distribution': 'uniform', 'min':args.adam_epsilon/2, 'max':args.adam_epsilon*2}})
    sweep_configuration['parameters'].update({'adam_epsilon':{'distribution': 'categorical', 'values':[args.adam_epsilon]}})# args.adam_epsilon, args.adam_epsilon*2 
    # sweep_configuration['parameters'].update({'weight_decay':{'distribution': 'uniform', 'min':0, 'max':0.2}})
    sweep_configuration['parameters'].update({'weight_decay':{'distribution': 'categorical', 'values':[args.weight_decay]}})# args.weight_decay, 0.1
    # sweep_configuration['parameters'].update({'warmup_ratio':{'distribution': 'uniform', 'min':0, 'max':0.2}})  
    sweep_configuration['parameters'].update({'warmup_ratio':{'distribution': 'categorical', 'values':[args.warmup_ratio]}})# args.warmup_ratio, args.warmup_ratio*2 
    # sweep_configuration['parameters'].update({'lr_scheduler':{'distribution': 'categorical', 'values':['linear']}})# 'cosine', 'cosine_w_restarts', 'linear', 'polynomial'
    # sweep_configuration['parameters'].update({'learning_rate':{'distribution': 'uniform', 'min':args.learning_rate/2, 'max':args.learning_rate*2}})
    sweep_configuration['parameters'].update({'learning_rate':{'distribution': 'categorical', 'values':[args.learning_rate]}})# args.learning_rate, args.learning_rate*2

    # batch size와 max_seq_length >> roberata large의 경우 GPU Memory 오류 발생 << 일정한 범위내로 제한 필요
    # sweep_configuration['parameters'].update({'train_batch_size':{'distribution': 'int_uniform', 'min':args.train_batch_size/2, 'max':args.train_batch_size+1}})
    # sweep_configuration['parameters'].update({'max_seq_length':{'distribution': 'int_uniform', 'min':args.max_seq_length/2, 'max':args.max_seq_length+1}})
    # sweep_configuration['parameters'].update({'max_seq_length':{'distribution': 'categorical', 'values':[args.max_seq_length]}})
    
    # 일부 주석처리: 모델별로 값이 다르게 들어가야 해서 우선 pretrained 된 config 기본 값들이 학습시 전달되도록 수정
    # hidden_size (int, optional, defaults to 768)
    #  — Dimensionality of the encoder layers and the pooler layer.
    # !!! ValueError('The hidden size is a multiple of the number of attention heads (12)')
    # sweep_configuration['parameters'].update({'hidden_size':{'distribution': 'constant', 'value':768}})# 이미 pretrained 된 가중치 값을 사용하기위해 기본값 그대로 사용해야 함
    # num_hidden_layers (int, optional, defaults to 12)
    #  — Number of hidden layers in the Transformer encoder.
    # sweep_configuration['parameters'].update({'num_hidden_layers':{'distribution': 'constant', 'value':12}})# 이미 pretrained 된 가중치 값을 사용하기위해 기본값 그대로 사용해야 함
    # num_attention_heads (int, optional, defaults to 12)
    #  — Number of attention heads for each attention layer in the Transformer encoder.
    # sweep_configuration['parameters'].update({'num_attention_heads':{'distribution': 'constant', 'value':12}})
    # intermediate_size (int, optional, defaults to 3072)
    #  — Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
    # sweep_configuration['parameters'].update({'intermediate_size':{'distribution': 'constant', 'value':3072}})
    # hidden_act (str or Callable, optional, defaults to "gelu")
    #  — The non-linear activation function (function or string) in the encoder and pooler. If string, "gelu", "relu", "silu" and "gelu_new" are supported.
    # sweep_configuration['parameters'].update({'hidden_act': {'distribution': 'categorical', 'values': ["gelu", "gelu_new"]}})
    # hidden_dropout_prob (float, optional, defaults to 0.1)
    #  — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    # sweep_configuration['parameters'].update({'hidden_dropout_prob':{'distribution': 'uniform', 'min':0, 'max':0.3}})
    # attention_probs_dropout_prob (float, optional, defaults to 0.1) 
    # — The dropout ratio for the attention probabilities.
    # sweep_configuration['parameters'].update({'attention_probs_dropout_prob':{'distribution': 'uniform', 'min':0, 'max':0.5}})
    # max_position_embeddings (int, optional, defaults to 512) 
    # — The maximum sequence length that this model might ever be used with. 
    # Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
    # sweep_configuration['parameters'].update({'max_position_embeddings':{'distribution': 'constant', 'value':514}})
    # type_vocab_size (int, optional, defaults to 2) 
    # — The vocabulary size of the token_type_ids passed when calling BertModel or TFBertModel.
    # sweep_configuration['parameters'].update({'type_vocab_size':{'distribution': 'constant', 'value':1}})# 이미 pretrained 된 가중치 값을 사용하기위해 기본값 그대로 사용해야 함
    # initializer_range (float, optional, defaults to 0.02)
    #  — The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    # sweep_configuration['parameters'].update({'initializer_range':{'distribution': 'categorical', 'values':[0.02, 0.03, 0.01]}})
    # layer_norm_eps (float, optional, defaults to 1e-12) 
    # — The epsilon used by the layer normalization layers.
    # sweep_configuration['parameters'].update({'layer_norm_eps':{'distribution': 'uniform', 'min':1e-12, 'max':1e-04}})
    # position_embedding_type (str, optional, defaults to "absolute")
    #  — Type of position embedding. Choose one of "absolute", "relative_key", "relative_key_query".
    #  For positional embeddings use "absolute".
    #  For more information on "relative_key", please refer to Self-Attention with Relative Position Representations (Shaw et al.).
    #  For more information on "relative_key_query", please refer to Method 4 in Improve Transformer Models with Better Relative Position Embeddings (Huang et al.).
    # sweep_configuration['parameters'].update({'position_embedding_type':{'distribution': 'categorical', 'values':['absolute']}})
    # classifier_dropout (float, optional) — The dropout ratio for the classification head.
    # sweep_configuration['parameters'].update({'classifier_dropout':{'distribution': 'uniform', 'min':0, 'max':0.5}})
    
    # wandb.config.update(args) # adds all of the arguments as config variables
    sweep_id = wandb.sweep(sweep_configuration)

    # run the sweep
    wandb.agent(sweep_id, function=my_train_func)
    



if __name__ == "__main__":
    main()
