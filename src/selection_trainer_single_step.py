"""
This file contains the trainer class for the generation of Chain of Thought explanations.
"""

import logging
import itertools
import json
import os
import re
import math
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
from iteration_utilities import deepflatten, flatten
from typing import List, Tuple, Callable, Iterable

from datasets import load_metric
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    get_scheduler,
)

import wandb

from src.dataloader import PreprocessDataset
from src.common.config import DataArguments, ModelArguments, TrainingArguments

logger = logging.getLogger(__name__)

wandb.init(
    project=TrainingArguments.project_name,
    config={"log": "all"}
)

wandb.run.name = TrainingArguments.output_dir.split("/")[-1]


class ExplanationTreeGenerator:
    """
    class to generate explanation tree
    here we use the model to select the factoids from the input
    and these are used to predict the intermediate conclusions
    """
    def __init__(
        self,
        data_args,
        model_args,
        train_args,
    ):
        self.data_args = data_args
        self.model_args = model_args
        self.train_args = train_args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        config = AutoConfig.from_pretrained(self.model_args.model_name_or_path, cache_dir=self.model_args.cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path, cache_dir=self.model_args.cache_dir, use_fast=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_args.model_name_or_path, config=config, cache_dir=self.model_args.cache_dir
        )
        self.model.to(self.device)

        self.train_set, self.val_set = self.load_data()

    def load_data(self):
        dataloader = PreprocessDataset(
            data_path=DataArguments.data_path,
            seed=DataArguments.seed,
            shuffle=DataArguments.shuffle,
        )
        train_set, val_set = dataloader.preprocess_data()
        # select train and val set where exp_chain has length==1
        # train_set = train_set.filter(lambda example: len(example["explanation_chain"])==1)
        # val_set = val_set.filter(lambda example: len(example["explanation_chain"])==1)
        return train_set, val_set

    def generate_base_input(self, prefix, _question, _answer, _hypothesis):
        input = f"{prefix} question: {_question.lstrip()} answer: {_answer.lstrip()} hypothesis: {_hypothesis.lstrip()}"
        return input

    def _encode_inputs(self, tokenizer, inputs, targets):

        """
        Create input prompts for the model
        :param examples: sample from dataset
        :return: encoded data
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_inputs = self.tokenizer(
            inputs,
            max_length=ModelArguments.max_src_length,
            padding=ModelArguments.padding,
            truncation=True,
            return_tensors="pt",
        )

        # Setup the tokenizer for targets
        if targets is not None:
            labels = self.tokenizer(
                text_target=targets,
                max_length=ModelArguments.max_target_length,
                padding=ModelArguments.padding,
                truncation=True,
                return_tensors="pt",
            )
            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss
            if ModelArguments.padding == "max_length":
                labels["input_ids"] = labels["input_ids"].masked_fill(
                    labels["input_ids"] == tokenizer.pad_token_id, -100)

            model_inputs["labels"] = labels["input_ids"]
        # ensure tensors are on the right device
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        # print(model_inputs)
        return model_inputs

    def _preprocess_factoid_generation(
        self,
        examples,
    ) -> Tuple[List[str], List[str]]:
        question = examples["question"]
        context = examples["context"]
        answer = examples["answer"]
        hypotheses = examples["hypothesis"]
        meta_info = examples["meta"]

        def generate_target_input(factoids):
            # replace none with empty string and joining with ;
            factoids = [factoid if factoid else "" for factoid in factoids]
            return ";".join(factoids)

        inputs = [
            self.generate_base_input(self.train_args.prefix, ques, ans, hypothesis)
            for ques, ans, hypothesis in zip(question, answer, hypotheses)
        ]
        # print(inputs)
        targets = [
            # join the factoid with ; to make it a single string
            generate_target_input(list(meta["triples"].values()))
            for meta in meta_info
        ]
        # print(targets)
        return inputs, targets

    def _preprocess_factoid_selection(self, examples):
        """
        Preprocess the data for intermediate explanation generation
        """
        # print("Preprocessing data for factoid selection...")
        # print("Examples: ", examples)
        proof_steps = examples["length_of_proof"]
        max_steps = max(proof_steps) + 1
        # overriden the max_steps
        max_steps = 1

        def generate_selection_prompt(prefix, factoids, question):
            # arrange one factoid per line with prefix sent 0, sent 1, sent 2, ...
            # separate factoid attached with ; with a new line
            # factoids = factoids[0].split(";")
            # factoids = [factoid if factoid else "" for factoid in factoids]
            factoids = [
                f"sent {i}: {factoid}" for i, factoid in enumerate(factoids)
            ]
            # append the prefix at start and question at the end
            prompt = [prefix] + factoids + [f"question: {ques}" if ques.endswith("?") else f"question: {ques}?"
                                              for ques in question]
            # join the factoids with new line
            prompt = "\n".join(prompt)
            return prompt


        def generate_target_input(selected_factoids):

            starting_fact = selected_factoids[0]
            if len(selected_factoids) == 1:
                targets = f"{starting_fact}."
                return targets

            # add prefix "We know that" to the second factoid and add "and" at the end if there are more than 2 factoids
            if len(selected_factoids) < 3:
                targets = f"{starting_fact}. We know that {selected_factoids[1]}."
            else:
                secondary_facts = [fact+' and' for fact in selected_factoids[2:]]
                targets = f"{starting_fact}. We know that {selected_factoids[1]} and {secondary_facts[0].rstrip(' and')}."
            # print("Targets: ", targets)
            # targets = [f"We know that {fact}." for fact in selected_factoids[1:]]
            # print("targets: ", targets)
            return targets

        def decrement(match):
            number = int(match.group(0))
            return " " + str(number - 1)

        base_inputs, inputs, targets = [], [], []

        model_outputs = []
        term_indices = []
        loss = 0
        remaining_factoids = []
        for proof_step in range(max_steps):
            context = examples["context"][0]
            # print("Context: ", context)
            sentences = context.split("sent")
            sentences = [sentences[i][3:] for i in range(len(sentences)) if i != 0]
            # print("Sentences: ", sentences)
            question = examples["question"]
            hypotheses = examples["hypothesis"]
            meta_info = examples["meta"]
            explanation_chain = examples["explanation_chain"]
            # print("explanation_chain: ", explanation_chain)
            predicted_factoids = []
            # at step 0, we generate the all factoids if generate_factoids is True
            # and filter the factoids that are required for the proof based on the explanation chain
            if proof_step == 0:
                # gold inputs from factoid generation step
                prev_inputs, prev_targets = self._preprocess_factoid_generation(examples)
                if self.train_args.generate_factoids:
                    logger.info("Generating factoids")
                    # generate factoids
                    model_inputs = self._encode_inputs(self.tokenizer, prev_inputs, prev_targets)
                    loss += self.model(**model_inputs)[0]
                    predicted_factoids = self.predict(model_inputs)
                    prev_targets = predicted_factoids
                    predicted_factoids = list(
                        deepflatten([factoid.split(";") for factoid in predicted_factoids], depth=1))
                    # print("predicted_factoids: ", predicted_factoids)
                # set prev_inputs to base_inputs to be used in other steps
                base_inputs = prev_inputs
                # print("base_inputs: ", base_inputs)

                # concatenate the previous inputs and targets
                # remove the trailing ; from the targets
                # NOTE: all factoids are used here
                prev_targets = [target.rstrip(";") for target in prev_targets]
                # print("prev_targets: ", prev_targets)
                inputs = [generate_selection_prompt(self.train_args.prefix, sentences, question)]
                # print("Inputs at step 0: ", inputs)

                # targets will be the gold factoids from the explanation chain
                sent_idx, int_idx = 0, 1
                selected_factoids = [
                    explanation[proof_step][sent_idx]
                    for explanation in explanation_chain
                ]

                selected_factoids = list(
                    deepflatten(selected_factoids, depth=1))
                # print("selected_factoids: ", selected_factoids)

                selected_factoids = [
                    re.sub(r'\d+', decrement, factoid) for factoid in selected_factoids
                ]
                # print("selected_factoids: ", selected_factoids)

                targets = [generate_target_input(selected_factoids)]

        # print("inputs: ", inputs)
        # print("targets: ", targets)
        # print("-" * 100)
        return inputs, targets

    def get_prediction(self, model, inputs):
        """
        Get prediction
        """
        model.eval()
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=ModelArguments.max_target_length,
            num_beams=ModelArguments.num_beams,
            repetition_penalty=ModelArguments.repetition_penalty,
            length_penalty=ModelArguments.length_penalty,
            early_stopping=True,
        )

        predicted_text = [self.tokenizer.decode(
            g, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ) for g in generated_ids]
        return predicted_text



    def _preprocess_pipeline(self, examples):
        # print("Preprocessing pipeline")
        # task 1: Generate and select factoids
        # task 2 : Generate intermediate conclusion
        # task 3 : Generate final conclusion
        inputs, targets = self._preprocess_factoid_selection(examples)
        # print("inputs: ", inputs)
        # print("targets: ", targets)
        model_inputs = self._encode_inputs(self.tokenizer, inputs, targets)
        return model_inputs

    def data_collator(self, features):
        input_ids = [x["input_ids"] for x in features]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = [x["attention_mask"] for x in features]
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = [x["labels"] for x in features]
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def predict(self, model_inputs):
        """
        Predicts the output for the given input.
        :param model_inputs: input to the model
        :return: output of the model
        """

        self.model.eval()
        # Generate output
        generated_ids = self.model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_length=self.model_args.max_target_length,
            num_beams=self.model_args.num_beams,
            repetition_penalty=self.model_args.repetition_penalty,
            length_penalty=self.model_args.length_penalty,
            early_stopping=True,
        )
        preds = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                 for g in generated_ids]
        return preds

    def train(self):

        if self.train_args.do_train:
            logger.info("*** Train ***")
            if self.train_args.max_train_samples is not None:
                # We will select sample from whole data if argument is specified
                max_train_samples = min(len(self.train_set), self.train_args.max_train_samples)
                self.train_set = self.train_set.select(range(max_train_samples))

            # set task prefix
            self.train_args.prefix = "Select facts:"
            train_dataset = self.train_set.map(
                self._preprocess_pipeline,
                batched=True,
                batch_size=self.data_args.batch_size,
                desc="Running tokenizer on train dataset",
            )

            if self.train_args.do_eval or self.train_args.do_predict:
                if self.train_args.max_val_samples is not None:
                    # We will select sample from whole data if argument is specified
                    max_val_samples = min(len(self.val_set), self.train_args.max_val_samples)
                    self.val_set = self.val_set.select(range(max_val_samples))
                eval_dataset = self.val_set.map(
                    self._preprocess_pipeline,
                    batched=True,
                    batch_size=self.data_args.batch_size,
                    desc="Running tokenizer on validation dataset",
                )

            args = Seq2SeqTrainingArguments(
                output_dir=self.train_args.output_dir,
                overwrite_output_dir=False,
                do_train=self.train_args.do_train,
                do_eval=self.train_args.do_eval,
                per_device_train_batch_size=self.train_args.batch_size,
                per_device_eval_batch_size=self.train_args.batch_size,
                learning_rate=self.train_args.learning_rate,
                num_train_epochs=self.train_args.epochs,
                weight_decay=self.train_args.weight_decay,
                lr_scheduler_type=self.train_args.lr_scheduler_type,
                save_strategy='epoch',
                evaluation_strategy='epoch',
                logging_strategy='steps',
                logging_steps=50,
                predict_with_generate=True,
                load_best_model_at_end=True,
                save_total_limit=2,
                disable_tqdm=False,
                report_to=["wandb"],
                remove_unused_columns=False,
                seed=self.train_args.seed,
                label_names=["labels"],  # it's important to log eval_loss
            )

            trainer = Trainer(
                model=self.model,
                args=args,
                data_collator=self.data_collator,
                train_dataset=train_dataset if self.train_args.do_train else None,
                eval_dataset=eval_dataset if self.train_args.do_eval else None,
            )
            try:
                if self.train_args.do_train:
                    checkpoint = None
                    if self.train_args.resume_from_checkpoint is not None:
                        checkpoint = self.train_args.resume_from_checkpoint
                    trainer.train(resume_from_checkpoint=checkpoint)
                    trainer.save_model()
            except KeyboardInterrupt:
                trainer.save_model("interrupted-exp-tree")

            wandb.finish()


if __name__ == "__main__":
    data_args = DataArguments()
    model_args = ModelArguments()
    training_args = TrainingArguments()
    trainer = ExplanationTreeGenerator(data_args, model_args, training_args)
    trainer.train()
