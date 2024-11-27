import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union, List
import torch
import torch.nn.functional as F

import numpy as np

from transformers import AutoConfig, AutoTokenizer, EvalPrediction, PreTrainedTokenizerBase
from src.modeling_roberta import RobertaConfig
from src.modeling_opt import OPTConfig
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed

from src.prefix import PrefixTuning
from src.dataset import FewShotDataset, OurInputFeatures
from src.models import MODEL_TYPES, resize_token_type_embeddings, convert_opt_model
from src.trainer import Trainer
from src.prune import model_pruning_with_data_free
from src.processors import num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping

from filelock import FileLock
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class ModelArguments:
    """
    Arguments for model/config/tokenizer/pruning
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # Only for BERT-type model
    random_segment: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )
    #pruning hyperparameters
    query_nums: int = field(
        default=1,
        metadata={"help": "The number of evaluate gradients"},
    )
    with_pruning: bool = field(
        default=False,
        metadata={"help": "Whether to use pruning"}
    )
    pruning_ratio: float = field(
        default=0.01,
        metadata={"help": "The ratio of parameters to optimize"}
    )
    zero_order_eps: float = field(
        default=1e-3,
        metadata={"help": 'eps for zero order pruning'}
    )
    pruning_mode: str = field(
        default="normal",
        metadata={"help": "Pick from {normal, rank}"}
    )
    upper_bound: float = field(
        default=1.2,
        metadata={"help": "Upper bound for rank-based pruning"}
    )
    lower_bound: float = field(
        default=0.8,
        metadata={"help": "Lower bound for rank-based pruning"}
    )

@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for data
    """
    num_k: Optional[int] = field(
        default=512,
        metadata={"help": "Number of training examples per class"}
    )
    num_sample: Optional[int] = field(
        default=16,
        metadata={"help": "Number of samples (for inference) in fine-tuning"}
    )
    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )
    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )
    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )
    
@dataclass
class DynamicTrainingArguments(TrainingArguments):
    """
    Arguments for training
    """
    evaluate_during_training: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation during training or at the."}
    )
    log_file: str = field(
        default='log'
    )
    
    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={"help": "Array ID (contains seed and hyper-parameter search) to idenfity the model"}
    )

    model_id: int = field(
        default=-1,
        metadata={"help": "Model ID (contains template information) to identify the model"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )
    
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )
    optimizer: str = field(
        default='adam',
        metadata={'help': 'choose sgd or adam. default is adam'}
    )
    optimizer_variant: str = field(
        default='',
        metadata={'help': 'define variants on optimizer: signgd'}
    )
    random_model_init: bool = field(
        default=False,
        metadata={'help': 'reinit the model randomly'}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Initial learning rate for finetuning"}
    )
    zero_order_optim: bool = field(
        default=False,
        metadata={'help': 'when on, trains the model by zero-order optimization'}
    )
    zero_order_clip_grad: bool = field(
        default=False,
        metadata={"help": "Clip the norm of the gradient for zero order (only when using trainer optimizer)"}
    )
    zero_order_eps_start: float = field(
        default=1e-6,
        metadata={'help': 'eps for the start of zero order optim'}
    )
    zero_order_eps_end: float = field(
        default=1e-6,
        metadata={'help': 'eps for the start of zero order optim'}
    )
    iteration_nums: int = field(
        default=1000,
        metadata={"help": "Initial iteration nums"}
    )
    stage_size: int = field(
        default=3,
        metadata={"help": "The size of stage for optimization"}
    )
    
    regularization: float = field(
        default=5e-4,
        metadata={"help": "Regularization coefficient during optimization process"}
    )
    
    perturb_nums: int = field(
        default=1,
        metadata={"help": "The number of perturb vectors"}
    )
    
    training_steps: int = field(
        default=6000,
        metadata={"help": "Total training steps"}
    )
        
    zero_order_use_trainer_optim: bool = field(
        default=False,
        metadata={"help": "Use trainer optimizer for zero order optimization"}
    )
    
    # prefix tuning hyperparameters
    prefix_tuning: bool = field(
        default=False,
        metadata={"help": "Prefix tuning"}
    )
    num_prefix: int = field(
        default=10,
        metadata={"help": "How many prefix tokens to use"}
    )
    no_reparam: bool = field(
        default=False,
        metadata={"help": "No reparameterization trick"}
    )
    prefix_init_by_real_act: bool = field(
        default=False,
        metadata={"help": "For no_reparam case, randomly sample words and take their actual key/value pairs as initialization"}
    )

    max_zo_forward_steps: int = field(
        default=0,
        metadata={'help': 'Stop at this number of ZO forward steps. The trainer will take whichever is reached first, max_steps or max_zo_forward_steps.'}
    )
    
    untie_emb: bool = field(
        default=False,
        metadata={"help": "Untie embeddings from lm head. Only work for OPT!!"}
    )
    tie_emb: bool = field(
        default=False,
        metadata={"help": "Tie embeddings from lm head. Only work for RoBERTa!!"}
    )
    
    optimize_acc: bool = field(
        default=False,
        metadata={"help": "Maximize accuracy instead of minimizing loss"}
    )
    
    # DP hyperparameters
    clip_threshold: float = field(
        default=30,
        metadata={"help": "The clip threshold for dp noise"}
    )
    
    epsilon: float = field(
        default=4,
        metadata={"help": "The privacy budget"}
    )
    
@dataclass
class MyDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        mask_pos = []
        standard_features = []

        for item in features:
            standard_item = {}
            for field in ["input_ids", "label", "attention_mask", "token_type_ids"]:
                if getattr(item, field) is not None:
                    standard_item[field] = getattr(item, field)
            standard_features.append(standard_item)
            mask_pos.append(item.mask_pos)

        batch = self.tokenizer.pad(
            standard_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if any(mask_pos):
            batch["mask_pos"] = torch.tensor(mask_pos)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
            
        return batch

def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    zo_mode = "zoso" if model_args.with_pruning else "zopo"
    training_args.output_dir = os.path.join(training_args.output_dir, zo_mode)
    
    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False
        
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")
    
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    
    # Set seed
    set_seed(training_args.seed)
    
    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))
    
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir
    )
    
    model_type = MODEL_TYPES[config.model_type]
    special_tokens = []
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir,
    )
    
    if "opt" in model_args.model_name_or_path:
        # Set SEP token
        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.bos_token_id = 0
    if "gpt2" in model_args.model_name_or_path:
        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = model_type.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    
    tokenizer.model_type = model.config.model_type

    if training_args.prefix_tuning:
        PrefixTuning(model, num_prefix=training_args.num_prefix, reparam=not training_args.no_reparam, init_by_real_act=training_args.prefix_init_by_real_act)
    
    # Get our special datasets.
    train_dataset = FewShotDataset(data_args, tokenizer=tokenizer, mode="train") if training_args.do_train else None
    eval_dataset = FewShotDataset(data_args, tokenizer=tokenizer, mode="dev") if training_args.do_eval else None
    test_dataset = FewShotDataset(data_args, tokenizer=tokenizer, mode="test") if training_args.do_predict else None
    
    set_seed(training_args.seed)
    
    if training_args.random_model_init:
        model.init_weights() # reinit weights to random

    # For BERT, increase the size of the segment (token type) embeddings
    if config.model_type == 'bert':
        model.resize_token_embeddings(len(tokenizer))
        resize_token_type_embeddings(model, new_num_types=10, random_segment=model_args.random_segment)
        
    # Pass dataset and argument information to the model
    if eval_dataset.label_word_list is not None:
        model.label_word_list = torch.tensor(eval_dataset.label_word_list).long().to(training_args.device)
    if output_modes_mapping[data_args.task_name] == 'regression':
        # lower / upper bounds
        model.lb, model.ub = bound_mapping[data_args.task_name]
    
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer
    model.to(training_args.device)
    
    # Build metric
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions
            num_logits = predictions.shape[-1]

            num_sample = test_dataset.num_sample if eval_dataset is None else eval_dataset.num_sample
            logits = predictions.reshape([num_sample, -1, num_logits])
            logits = logits.mean(axis=0)

            if num_logits == 1:
                preds = np.squeeze(logits)
            else:
                preds = np.argmax(logits, axis=1)

            # Just for sanity, assert label ids are the same.
            label_ids = p.label_ids.reshape([num_sample, -1])
            label_ids_avg = label_ids.mean(axis=0)
            label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
            assert (label_ids_avg - label_ids[0]).mean() < 1e-2
            label_ids = label_ids[0]

            return compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn
    
    # Pruning
    if model_args.with_pruning:
        mask = model_pruning_with_data_free(model_args, model)
    else:
        mask = None
    
    # Initialize our Trainer
    trainer_kwargs = {}
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        data_collator=MyDataCollatorWithPadding(tokenizer),
        **trainer_kwargs
    )
    
    # Training
    if training_args.do_train:
        trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None, mask=mask)
        
        if training_args.save_at_last:
            trainer.save_model(training_args.output_dir)

        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
            torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
            torch.save(data_args, os.path.join(training_args.output_dir, "data_args.bin"))
        
        if training_args.evaluate_during_training:
            trainer.model.load_state_dict(trainer.best_model_ckpt)
    
    # Evaluation
    final_result = {
        'time': str(datetime.today()),
        'output_dir': training_args.output_dir
    }
    
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Validate ***")

        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=eval_dataset)
            eval_result = output.metrics

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[eval_dataset.args.task_name + '_dev_' + key] = value
            eval_results.update(eval_result)

    test_results = {}
    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]

        for test_dataset in test_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=test_dataset)
            test_result = output.metrics

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    for key, value in test_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[test_dataset.args.task_name + '_test_' + key] = value

                if training_args.save_logit:
                    predictions = output.predictions
                    num_logits = predictions.shape[-1]
                    logits = predictions.reshape([test_dataset.num_sample, -1, num_logits]).mean(axis=0)
                    np.save(os.path.join(training_args.save_logit_dir, "{}-{}-{}.npy".format(test_dataset.task_name, training_args.model_id, training_args.array_id)), logits)

            test_results.update(test_result)


    if trainer.is_world_process_zero():
        with FileLock('log.lock'):
            with open(training_args.log_file, 'a') as f:
                final_result.update(vars(model_args))
                final_result.update(vars(training_args))
                final_result.update(vars(data_args))
                if 'evaluation_strategy' in final_result:
                    final_result.pop('evaluation_strategy')
                f.write(str(final_result) + '\n')

    logger.info('****** Output Dir *******')
    logger.info(training_args.output_dir)

    return eval_results

if __name__ == "__main__":
    main()
