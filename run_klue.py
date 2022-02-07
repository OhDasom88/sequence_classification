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
        default=None,# 20220202
        # default=[0],
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
        default=5,
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
            save_top_k=1,
            mode="max",
        )
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
    
    wandb.watch(task.model, criterion=None, log="all", log_freq=1000, idx=None, log_graph=(False))    

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
        "name": f"{args.model_name_or_path}",
        "metric": {"name": "valid/accuracy", "goal": "maximize"},
        # "metric": {"name": "valid/loss", "goal": "minimize"},
        "method": "grid",
        # "method": "bayes",
        "parameters": {},
        'early_terminate':{'type': 'hyperband' , 'max_iter': 27, 's': 2}

    }
    sweep_configuration['parameters'] = {k:{'values': [v]} for k , v in vars(args).items()}
    # sweep_configuration['parameters'].update({'hidden_dropout_prob':{'distribution': 'uniform', 'min':0, 'max':0.5}})
    # sweep_configuration['parameters'].update({'attention_probs_dropout_prob':{'distribution': 'uniform', 'min':0, 'max':0.5}})
    sweep_configuration['parameters'].update({'hidden_dropout_prob':{'values':[0.1,0.3, 0.5]}})
    sweep_configuration['parameters'].update({'attention_probs_dropout_prob':{'values':[0.1,0.3, 0.5]}})
    sweep_configuration['parameters'].update({'layer_norm_eps':{'values':[1e-05/2,1e-05,1e-05*2]}})
    sweep_configuration['parameters'].update({'weight_decay':{'values':[0.0,0.3, 0.5]}})
    sweep_configuration['parameters'].update({'hidden_act': {'values': ["gelu", "relu", "swish" , "gelu_new"]}})
    sweep_configuration['parameters'].update({'fp16':{'values':[True, False]}})
    sweep_configuration['parameters'].update({'train_batch_size':{'values':[args.train_batch_size/2, args.train_batch_size,args.train_batch_size*2]}})
    sweep_configuration['parameters'].update({'max_seq_length':{'values':[args.max_seq_length/2,args.max_seq_length,args.max_seq_length*2]}})
    sweep_configuration['parameters'].update({'adafactor':{'values':[True, False]}})
    sweep_configuration['parameters'].update({'adam_epsilon':{'values':[args.adam_epsilon/2, args.adam_epsilon,args.adam_epsilon*2]}})
    sweep_configuration['parameters'].update({'learning_rate':{'values':[args.learning_rate/2, args.learning_rate, args.learning_rate*2]}})
    
    # sweep_configuration['parameters'].update({'decoder_layerdrop':{'values':[0.1, 0.5]}})
    # sweep_configuration['parameters'].update({'dropout':{'values':[0.1, 0.5]}})
    # sweep_configuration['parameters'].update({'attention_dropout':{'values':[0.1, 0.5]}})
    # sweep_configuration['parameters'].update({'attention_probs_dropout_prob':{'values':[0.1, 0.5]}})
    # sweep_configuration['parameters'].update({'hidden_dropout_prob':{'values':[0.1, 0.5]}})
    # sweep_configuration['parameters'].update({'hidden_act':{'values':[False]}})

    # wandb.config.update(args) # adds all of the arguments as config variables
    sweep_id = wandb.sweep(sweep_configuration)

    # run the sweep
    wandb.agent(sweep_id, function=my_train_func)
    



if __name__ == "__main__":
    main()
