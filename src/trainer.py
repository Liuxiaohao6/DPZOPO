########## The following part is copied from Transformers' trainer (3.4.0) and later ported to be compatible with v4.4.2 and to support initialization from linear head probing. ##########

# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a ? Transformers from scratch or finetune it on a new task.
"""

import collections
import inspect
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import math
import time

import transformers
from transformers.file_utils import is_datasets_available, is_in_notebook, is_torch_tpu_available
from transformers.integrations import (
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_scheduler

from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_utils import (
    default_compute_objective,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import TrainOutput

from tqdm import tqdm, trange
from torch.optim import SGD
import torch.nn.functional as F
from prv_accountant import Accountant
from transformers.trainer_callback import TrainerState

import copy

_use_native_amp = False
_use_apex = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

########## The above part is copied from Transformers' trainer (3.4.0) ##########

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))

class Trainer(transformers.Trainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """

        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.optimizer == 'adam':
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif self.args.optimizer == 'sgd':
                self.optimizer = SGD(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate
                )
            else:
                raise NotImplementedError
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
            
    def get_eps(self, q, steps, delta, sigma):
        accountant = Accountant(
            noise_multiplier=sigma,
            sampling_probability=q,
            delta=delta,
            eps_error=0.1,
            max_compositions=steps)       
        eps_low, eps_estimate, eps_upper = accountant.compute_epsilon(num_compositions=steps)
        eps = eps_upper
        return eps
            
    def loop_for_sigma(self, q, steps, eps, delta, cur_sigma, interval):
        while True:
            cur_eps = self.get_eps(q, steps, delta, cur_sigma)
            if(cur_eps<eps and cur_sigma>interval):
                cur_sigma -= interval
                previous_eps = cur_eps
            else:
                cur_sigma += interval
                break    
        return cur_sigma, previous_eps        
    
    def get_sigma(self, q, T, eps, delta, init_sigma=10, interval=0.5):
        cur_sigma = init_sigma
        
        cur_sigma, _ = self.loop_for_sigma(q, T, eps, delta, cur_sigma, interval)
        interval /= 10
        cur_sigma, _ = self.loop_for_sigma(q, T, eps, delta, cur_sigma, interval)
        interval /= 10
        cur_sigma, _ = self.loop_for_sigma(q, T, eps, delta, cur_sigma, interval)
        interval /= 10
        cur_sigma, eps = self.loop_for_sigma(q, T, eps, delta, cur_sigma, interval)
        return cur_sigma, eps
    
    def zo_forward(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.eval()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss_ave, logits = model(**inputs)
            loss =[]
            for i in range(inputs["labels"].numel()):
                tem_loss = F.cross_entropy(logits[i],inputs["labels"][i]).to(device=logits.data.device)
                loss.append(tem_loss.item())
        self.state.zo_forward_step += 1
        return loss

    def vector_wise_perturb_parameters(self, model: nn.Module, random_seed: int, mask: dict, zero_order_eps: float, scaling_factor=1):
        torch.manual_seed(random_seed)
        for name, param in self.named_parameters_to_optim:
            if mask is not None:
                indices = [*mask[name].indices()]
                z = torch.normal(mean=0, std=1, size=(indices[-1].shape[0],), device=param.data.device, dtype=param.data.dtype)
                param[indices].data += scaling_factor * z * zero_order_eps * mask[name].values()
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data += scaling_factor * z * zero_order_eps
        return model
        

    def train(self, model_path=None, mask=None, dev_objective=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """
        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        # Data loading.
        train_dataloader = self.get_train_dataloader()
        max_steps = min(self.args.iteration_nums  * np.sum([2 ** i for i in range(self.args.stage_size)]), self.args.training_steps)
        num_train_epochs = math.ceil(max_steps / len(train_dataloader))
        num_examples = self.num_examples(train_dataloader)
        sigma, eps = self.get_sigma(q=self.args.train_batch_size/num_examples, T=max_steps, eps=self.args.epsilon, delta=1/num_examples)
        logger.info("simga=%f, eps=%f", sigma, eps)

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        # Train
        if transformers.is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", max_steps)

        self.state = TrainerState()
        self.state.global_step = 0
        self.state.zo_forward_step = 0
        self.epoch = 0
        epochs_trained = 0
        current_stage = 0
        zero_order_eps_k = np.power(self.args.zero_order_eps_end - self.args.zero_order_eps_start, 1 / self.args.stage_size) \
            if self.args.zero_order_eps_end != self.args.zero_order_eps_start else 1
        zero_order_eps = self.args.zero_order_eps_start

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        metrics = None
        prev_parameters = model.state_dict()
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

            for step, inputs in enumerate(epoch_iterator):
                sample_num = inputs['input_ids'].shape[0]
                if self.args.zero_order_optim:
                    self.named_parameters_to_optim = []
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            self.named_parameters_to_optim.append((name, param))

                    perturb_seed = torch.zeros(self.args.perturb_nums).to(dtype=torch.int)
                    perturb_grad = torch.zeros(self.args.perturb_nums)
                    for idx in range(self.args.perturb_nums):
                        random_seed = np.random.randint(1000000000)
                        with torch.no_grad():
                            # first function evaluation
                            model = self.vector_wise_perturb_parameters(model, random_seed, mask, zero_order_eps)
                            loss1 = self.zo_forward(model, inputs)
                            # second function evaluation
                            model = self.vector_wise_perturb_parameters(model, random_seed, mask, zero_order_eps, scaling_factor=-2)               
                            loss2 = self.zo_forward(model, inputs)
                            # reset model back to its parameters at start of step
                            model = self.vector_wise_perturb_parameters(model, random_seed, mask, zero_order_eps)        
                        grads = []
                        for sample_idx in range(sample_num):
                            sample_grad = (loss1[sample_idx] - loss2[sample_idx]) / (2 * zero_order_eps)
                            clip_sample_grad = sample_grad / max(1, abs(sample_grad) / self.args.clip_threshold)
                            grads.append(clip_sample_grad)
                        noise = np.random.normal(loc=0, scale=sigma * self.args.clip_threshold)
                        grad_sum = sum(grads) + noise
                        perturb_seed[idx] = random_seed
                        perturb_grad[idx] = grad_sum / sample_num
                    
                if self.args.zero_order_optim:
                    # store gradient in parameter buffer if using trainer
                    # o/w, the loop will exit after one round and the update will be applied directly (see below)
                    if self.args.zero_order_use_trainer_optim:
                        for idx in range(self.args.perturb_nums):
                            torch.manual_seed(perturb_seed[idx])
                            for name, param in self.named_parameters_to_optim:
                                if mask is not None:
                                    indices = [*mask[name].indices()]
                                    z = torch.normal(mean=0, std=1, size=(indices[-1].shape[0],), device=param.data.device, dtype=param.data.dtype)
                                    if param.grad is None:
                                        param.grad = 0
                                        param[indices].grad = (param[indices].data - prev_parameters[name][indices].data) * self.args.regularization
                                    param[indices].grad += perturb_grad[idx] * z / self.args.perturb_nums
                                else:
                                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                                    if param.grad is None:
                                        param.grad = 0
                                        param.grad = (param.data - prev_parameters[name].data) * self.args.regularization
                                    param.grad += perturb_grad[idx] * z / self.args.perturb_nums

                    # apply gradient updates
                    # if using trainer, follow trainer logic to clip grad and check if parameters should be updated
                    if self.args.zero_order_use_trainer_optim:
                        # Gradient norm clipping
                        if self.args.zero_order_clip_grad:
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_threshold)

                        # Update the parameters and step scheduler
                        optimizer.step()
                        scheduler.step()
                    
                        # logging
                        if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                            self.state.global_step == 1 and self.args.logging_first_step
                        ):
                            logs = {}
                            logs["loss"] = loss1.item()
                            if not self.args.zero_order_clip_grad:
                                norm = 0.0
                                for _, p in model.named_parameters():
                                    if p.grad is not None:
                                        norm += torch.sum(p.grad ** 2)
                                norm = torch.sqrt(norm)
                            logs["grad_norm"] = norm.item()
                            logs["learning_rate"] = (
                                scheduler.get_last_lr()[0]
                                if version.parse(torch.__version__) >= version.parse("1.4")
                                else scheduler.get_lr()[0]
                            )
                            logs["global_step"] = self.state.global_step
                            logs["zo_forward_step"] = self.state.zo_forward_step
                            logs["max_steps"] = max_steps
                            self.log(logs)
                            logger.info(str(logs))
                        
                        model.zero_grad()
                        self.state.global_step += 1
                        self.epoch = epoch + (step + 1) / len(epoch_iterator)
                    # if not using the trainer, the updates are resampled and directly applied to the parameters
                    else: 
                        for idx in range(self.args.perturb_nums):
                            torch.manual_seed(perturb_seed[idx])
                            for name, param in self.named_parameters_to_optim:
                                if mask is not None:
                                    indices = [*mask[name].indices()]
                                    z = torch.normal(mean=0, std=1, size=(indices[-1].shape[0],), device=param.data.device, dtype=param.data.dtype)
                                    regular_param = (param[indices].data - prev_parameters[name][indices].data) * self.args.regularization
                                    regular_grad = perturb_grad[idx] + regular_param
                                    param[indices].data -= self.args.learning_rate * regular_grad * z / self.args.perturb_nums
                                else:
                                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                                    regular_param = (param.data - prev_parameters[name].data) * self.args.regularization
                                    regular_grad = perturb_grad[idx] + regular_param
                                    param.data -= self.args.learning_rate * regular_grad * z / self.args.perturb_nums

                        if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                                self.state.global_step == 1 and self.args.logging_first_step
                            ):
                                logs = {}
                                logs["learning_rate"] = self.args.learning_rate
                                logs["global_step"] = self.state.global_step
                                logs["zo_forward_step"] = self.state.zo_forward_step
                                logs["max_steps"] = max_steps
                                self.log(logs)
                                logger.info(str(logs))


                        self.state.global_step += 1
                        self.epoch = epoch + (step + 1) / len(epoch_iterator)

                # standard, non-ZO optimization
                else:
                    tr_loss += self.training_step(model, inputs)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    
                    noise = np.random.normal(loc=0, scale=sigma)
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad += noise / sample_num

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    
                    self.state.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                        self.state.global_step == 1 and self.args.logging_first_step
                    ):
                        logs = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        logs["norm"] = norm.item()
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar

                        self.log(logs)
                        logger.info(str(logs))
                
                if self.state.global_step > self.args.iteration_nums  * np.sum([2 ** i for i in range(current_stage + 1)]):
                    current_stage += 1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    self.args.learning_rate *= 0.5
                    zero_order_eps /= zero_order_eps_k
                    prev_parameters = model.state_dict()
                        
                if self.state.global_step > max_steps:
                    epoch_iterator.close()
                    break

                if self.args.evaluate_during_training and self.state.global_step % self.args.eval_steps == 0:
                    output = self.evaluate()
                    metrics = output.metrics
                    objective = self.dev_objective(metrics)
                    if objective > self.objective:
                        logger.info("Best dev result: {}".format(objective))
                        self.objective = objective
                        # self.save_model(self.args.output_dir)

                        # Now we save this to (CPU) memory instead of disk <-- much faster
                        self.best_model_ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            if self.state.global_step > max_steps:
                break
            
        output = self.evaluate()
        metrics = output.metrics
        objective = self.dev_objective(metrics)
        self.objective = objective
        self.best_model_ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.state.global_step, tr_loss / self.state.global_step, metrics), self.objective


    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)
        logger.info(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output
