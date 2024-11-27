"""Dataset utils for different data settings for GLUE."""

import os
import logging
import torch
import numpy as np
import time
from filelock import FileLock
import json
from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, median_mapping
from transformers.data.processors.utils import InputFeatures
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass(frozen=True)
class OurInputFeatures(InputFeatures):
    """
    Inherit from Transformers' InputFeatuers.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    mask_pos: Optional[List[int]] = None # Position of the mask token
    label_word_list: Optional[List[int]] = None # Label word mapping (dynamic)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

def input_example_to_string(example, sep_token):
    if example.text_b is None:
        return example.text_a
    else:
        # Warning: very simple hack here
        return example.text_a + ' ' + sep_token + ' ' + example.text_b

def input_example_to_tuple(example):
    if example.text_b is None:
        if pd.isna(example.text_a) or example.text_a is None:
            return ['']
            logger.warn("Empty input")
        else:
            return [example.text_a]
    else:
        return [example.text_a, example.text_b]

def tokenize_multipart_input(
    input_text_list,
    max_length,
    tokenizer,
    template=None,
    label_word_list=None,
    first_sent_limit=None,
    other_sent_limit=None,
    truncate_head=False,
    support_labels=None,
):
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    input_ids = []
    attention_mask = []
    token_type_ids = [] # Only for BERT
    mask_pos = None # Position of the mask token

    """
    Concatenate all sentences and prompts based on the provided template.
    Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
    *xx* represent variables:
        *cls*: cls_token
        *mask*: mask_token
        *sep*: sep_token
        *sep+*: sep_token, also means +1 for segment id
        *sent_i*: sentence i (input_text_list[i])
        *sent-_i*: same as above, but delete the last token
        *sentl_i*: same as above, but use lower case for the first word
        *sentl-_i*: same as above, but use lower case for the first word and delete the last token
        *+sent_i*: same as above, but add a space before the sentence
        *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
        *label_i*: label_word_list[i]
        *label_x*: label depends on the example id (support_labels needed). this is only used in GPT-3's in-context learning

    Use "_" to replace space.
    PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
    """
    assert template is not None

    special_token_mapping = {
        'bos': tokenizer.bos_token_id, 'cls': tokenizer.cls_token_id, 'eos': tokenizer.eos_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id, 'sep+': tokenizer.sep_token_id,
    }
    template_list = template.split('*') # Get variable list in the template
    segment_id = 0 # Current segment id. Segment id +1 if encountering sep+.

    for part_id, part in enumerate(template_list):
        new_tokens = []
        segment_plus_1_flag = False
        if part in special_token_mapping:
            if (part == 'cls' or part == 'bos') and ('T5' in type(tokenizer).__name__ or tokenizer.model_type == "gpt2"):
                # T5 or GPT-2 do not have cls token
                continue
            new_tokens.append(special_token_mapping[part])
            if part == 'sep+':
                segment_plus_1_flag = True
        elif part[:6] == 'label_':
            # Note that label_word_list already has extra space, so do not add more space ahead of it.
            label_id = int(part.split('_')[1])
            label_word = label_word_list[label_id]
            new_tokens.append(label_word)
        elif part[:7] == 'labelx_':
            instance_id = int(part.split('_')[1])
            label_id = support_labels[instance_id]
            label_word = label_word_list[label_id]
            new_tokens.append(label_word)
        elif part[:5] == 'sent_':
            sent_id = int(part.split('_')[1])
            new_tokens += enc(input_text_list[sent_id])
        elif part[:6] == '+sent_':
            # Add space
            sent_id = int(part.split('_')[1])
            new_tokens += enc(' ' + input_text_list[sent_id])
        elif part[:6] == 'sent-_':
            # Delete the last token
            sent_id = int(part.split('_')[1])
            new_tokens += enc(input_text_list[sent_id][:-1])
        elif part[:6] == 'sentl_':
            # Lower case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(text)
        elif part[:7] == '+sentl_':
            # Lower case the first token and add space
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(' ' + text)
        elif part[:7] == 'sentl-_':
            # Lower case the first token and discard the last token
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(text[:-1])
        elif part[:6] == 'sentu_':
            # Upper case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].upper() + text[1:]
            new_tokens += enc(text)
        elif part[:7] == '+sentu_':
            # Upper case the first token and add space
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].upper() + text[1:]
            new_tokens += enc(' ' + text)
        elif part[:8] == '+sentu-_':
            # Upper case the first token and add space
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].upper() + text[1:]
            new_tokens += enc(' ' + text[:-1])
        else:
            # Just natural language prompt
            part = part.replace('_', ' ')
            # handle special case when T5 tokenizer might add an extra space
            if len(part) == 1:
                new_tokens.append(tokenizer.convert_tokens_to_ids(part))
            else:
                new_tokens += enc(part)

        if part[:4] == 'sent' or part[1:5] == 'sent':
            # If this part is the sentence, limit the sentence length
            sent_id = int(part.split('_')[1])
            if sent_id == 0:
                if first_sent_limit is not None:
                    new_tokens = new_tokens[:first_sent_limit]
            else:
                if other_sent_limit is not None:
                    new_tokens = new_tokens[:other_sent_limit]

        input_ids += new_tokens
        attention_mask += [1 for i in range(len(new_tokens))]
        token_type_ids += [segment_id for i in range(len(new_tokens))]

        if segment_plus_1_flag:
            segment_id += 1

    # Padding
    if first_sent_limit is not None and len(input_ids) > max_length:
        # If using sentence limit, the total length still exceeds the maximum limit, report a warning
        logger.warn("Input exceeds max_length limit: {}".format(tokenizer.decode(input_ids)))

    # Truncate
    if len(input_ids) > max_length:
        if truncate_head:
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            token_type_ids = token_type_ids[-max_length:]
        else:
            # Default is to truncate the tail
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

    # Find mask token
    if tokenizer.mask_token_id is not None:
        # Make sure that the masked position is inside the max_length
        assert tokenizer.mask_token_id in input_ids, \
            "Mask token not found for input: {} {}".format(input_text_list, input_ids)
        mask_pos = [input_ids.index(tokenizer.mask_token_id)]
        assert mask_pos[0] < max_length
    elif tokenizer.mask_token_id is None:
        # autoregressive model
        mask_pos = [len(input_ids) - 1]

    result = {'input_ids': input_ids, 'attention_mask': attention_mask}
    if 'BERT' in type(tokenizer).__name__:
        # Only provide token type ids for BERT
        result['token_type_ids'] = token_type_ids

    result['mask_pos'] = mask_pos

    return result


class FewShotDataset(torch.utils.data.Dataset):
    """Few-shot dataset."""

    def __init__(self, args, tokenizer, cache_dir=None, mode="train"):
        self.args = args
        self.task_name = args.task_name

        self.processor = processors_mapping[args.task_name]

        self.tokenizer = tokenizer
        self.mode = mode

        # Get label list and (for prompt) label word list
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        if args.mapping is not None:
            self.label_to_word = eval(args.mapping)

            for key in self.label_to_word:
                # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
                if self.label_to_word[key][0] not in ['<', '[', '.', ',']:
                    # Make sure space+word is in the vocabulary
                    assert len(tokenizer.tokenize(' ' + self.label_to_word[key])) == 1
                    self.label_to_word[key] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + self.label_to_word[key])[0])
                else:
                    self.label_to_word[key] = tokenizer.convert_tokens_to_ids(self.label_to_word[key])
                logger.info("Label {} to word {} ({})".format(key, tokenizer.convert_ids_to_tokens(self.label_to_word[key]), self.label_to_word[key]))

            if len(self.label_list) > 1:
                self.label_word_list = [self.label_to_word[label] for label in self.label_list]
            else:
                # Regression task
                # '0' represents low polarity and '1' represents high polarity.
                self.label_word_list = [self.label_to_word[label] for label in ['0', '1']]
        else:
            self.label_to_word = None
            self.label_word_list = None

        self.num_sample = 1
        logger.info("Total num_sample for mode %s: %d" % (mode, self.num_sample))
        
        # Load cache
        # Cache name distinguishes mode, task name, tokenizer, and length. So if you change anything beyond these elements, make sure to clear your cache.
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode,
                tokenizer.__class__.__name__ + "-" + tokenizer.model_type,
                str(args.max_seq_length),
                args.task_name,
            ),
        )

        logger.info(f"Creating/loading examples from dataset file at {args.data_dir}")

        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.query_examples = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                
                if mode == "train":
                    self.query_examples = self.processor.get_train_examples(args.data_dir)
                elif mode == "test":
                    self.query_examples = self.processor.get_test_examples(args.data_dir)
                else:
                    self.query_examples = self.processor.get_dev_examples(args.data_dir)

                start = time.time()
                torch.save(self.query_examples, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

        # Size is expanded by num_sample
        self.size = len(self.query_examples) * self.num_sample

        # Prepare examples
        self.example_idx = []
        for _ in range(self.num_sample):
            for query_idx in range(len(self.query_examples)):
                self.example_idx.append(query_idx)

        # If it is not training, we pre-process the data; otherwise, we process the data online.
        if mode != "train":
            self.features = []
            _ = 0
            for query_idx in self.example_idx:
                # The input (query) example
                example = self.query_examples[query_idx]
                
                template = args.template
                self.features.append(self.convert_fn(
                    example=example,
                    label_list=self.label_list,
                    template=template,
                    label_word_list=self.label_word_list,
                    verbose=True if _ == 0 else False,
                ))

                _ += 1
        else:
            self.features = None

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.features is None:
            query_idx = self.example_idx[i]
            # The input (query) example
            example = self.query_examples[query_idx]

            template = self.args.template

            features = self.convert_fn(
                example=example,
                label_list=self.label_list,
                template=template,
                label_word_list=self.label_word_list,
                verbose=False,
            )

        else:
            features = self.features[i]

        return features

    def get_labels(self):
        return self.label_list


    def convert_fn(
        self,
        example,
        label_list=None,
        template=None,
        label_word_list=None,
        verbose=False
    ):
        """
        Returns a list of processed "InputFeatures".
        """
        max_length = self.args.max_seq_length

        # Prepare labels
        label_map = {label: i for i, label in enumerate(label_list)} # Mapping the label names to label ids
        if len(label_list) == 1:
            # Regression
            label_map = {'0': 0, '1': 1}

        # Get example's label id (for training/inference)
        if example.label is None:
            example_label = None
        elif len(label_list) == 1:
            # Regerssion
            example_label = float(example.label)
        else:
            example_label = label_map[example.label]

        inputs = tokenize_multipart_input(
            input_text_list=input_example_to_tuple(example),
            max_length=max_length,
            tokenizer=self.tokenizer,
            template=template,
            label_word_list=label_word_list,
            first_sent_limit=self.args.first_sent_limit,
            other_sent_limit=self.args.other_sent_limit,
        )
        features = OurInputFeatures(**inputs, label=example_label)

        if verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("features: %s" % features)
            logger.info("text: %s" % self.tokenizer.decode(features.input_ids))

        return features