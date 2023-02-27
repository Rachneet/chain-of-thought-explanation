"""
This file contains the trainer class for the generation of Chain of Thought explanations.
"""

import logging
import itertools
import json
import os
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

def load_data():
    dataloader = PreprocessDataset(
        data_path=DataArguments.data_path,
        seed=DataArguments.seed,
        shuffle=DataArguments.shuffle,
    )
    train_set, val_set = dataloader.preprocess_data()
    return train_set, val_set

TRAIN_DATA, VAL_DATA = load_data()


def generate_base_input(prefix, _question, _answer, _hypothesis):
    input = f"{prefix} question: {_question.lstrip()} answer: {_answer.lstrip()} hypothesis: {_hypothesis.lstrip()}"
    return input

def _encode_inputs(tokenizer, inputs, targets):

    """
    Create input prompts for the model
    :param examples: sample from dataset
    :return: encoded data
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_inputs = tokenizer(
        inputs,
        max_length=ModelArguments.max_src_length,
        padding=ModelArguments.padding,
        truncation=True,
        return_tensors="pt",
    )

    # Setup the tokenizer for targets
    if targets is not None:
        labels = tokenizer(
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


class ExplanationTrainer(Trainer):
    """
    Trainer class for explanation generation
    Overrides the training and evaluation methods
    in huggingface's Trainer class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            ModelArguments.model_name_or_path
        )
        self.train_data, self.val_data = TRAIN_DATA, VAL_DATA
        self.val_steps = 0

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

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss
        """

        # print("Computing loss...")
        loss = None
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        factoid_loss = outputs[0]

        # get current training step
        current_epoch = math.floor(self.state.epoch)
        num_train_examples = len(self.train_data) if TrainingArguments.max_train_samples is None \
            else TrainingArguments.max_train_samples
        # print("epoch: ", current_epoch)
        current_step = self.state.global_step - (num_train_examples * current_epoch)

        # Generate output
        predicted_text = self.get_prediction(model, inputs)
        # print("global step in com loss: ", self.state.global_step)
        intermediate_loss, outputs, _ = self.compute_intermediate_loss(
            self.train_data, current_step, return_outputs=True)
        loss = (factoid_loss + intermediate_loss)/2
        # print("Loss: ", loss)

        return (loss, outputs) if return_outputs else loss

    def compute_prediction_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss
        """

        current_epoch = math.floor(self.state.epoch)
        # print("epoch: ", current_epoch)
        num_val_examples = len(self.val_data) if TrainingArguments.max_val_samples is None \
            else TrainingArguments.max_val_samples
        current_step = self.val_steps - (num_val_examples * (current_epoch-1))

        loss = None
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        factoid_loss = outputs[0]

        # Generate output
        predicted_text = self.get_prediction(model, inputs)
        intermediate_loss, outputs, labels = self.compute_intermediate_loss(
            self.val_data, current_step, return_outputs=True)
        loss = (factoid_loss + intermediate_loss)/2
        # print("Loss: ", loss)
        self.val_steps += 1

        return (loss, outputs, labels) if return_outputs else loss

    def compute_intermediate_loss(self, inputs, step, return_outputs=False):
        intermediate_loss = 0
        intermediate_loss, outputs, labels = self._preprocess_intermediate_conclusion_generation(
            inputs[step])
        # print("Int. Loss: ", intermediate_loss)
        if return_outputs:
            return intermediate_loss, outputs, labels
        return intermediate_loss


    def _preprocess_intermediate_conclusion_generation(
        self,
        examples,
    ) -> Tuple[List[str], List[str], List]:
        # make every key in examples a list
        for key in examples.keys():
            examples[key] = [examples[key]]

        proof_steps = examples["length_of_proof"][0]

        TrainingArguments.prefix = "Predict intermediate conclusion:"
        inputs, targets = [], []
        base_inputs = [
            generate_base_input(TrainingArguments.prefix, ques, ans, hypothesis)
            for ques, ans, hypothesis in zip(examples["question"], examples["answer"], examples["hypothesis"])
        ]
        model_inputs: dict = {}
        model_outputs = []
        term_indices = []
        loss = 0
        remaining_factoids = []
        for proof_step in range(proof_steps):
            hypotheses = examples["hypothesis"]
            meta_info = examples["meta"]
            explanation_chain = examples["explanation_chain"]
            # print("explanation_chain: ", explanation_chain)
            predicted_factoids = []

            for base_input, meta_info, hypothesis, exp_chain in zip(
                    base_inputs, meta_info, hypotheses, explanation_chain):
                exp_step = exp_chain[proof_step]
                inp_idx, out_idx = 0, 1

                # outputs can have intermediate conclusions or the final conclusion (hypothesis)
                out = exp_step[out_idx]
                out_text = [meta_info["intermediate_conclusions"][x] if "int" in x else hypothesis for x in out]
                targets = [out_text for out_text in out_text]

                if TrainingArguments.use_gold_inputs:
                    inp = exp_step[inp_idx]
                    inp_text = [meta_info["triples"][x] if "sent" in x
                                else meta_info["intermediate_conclusions"][x] for x in inp]
                    inputs = [base_input + " " + f"facts: {';'.join(inp_text)}"]
                    # print("inp_text: ", inp_text)
                else:
                    # check if we are in the last step of hypothesis generation
                    # and if we need to use the generated factoids
                    if TrainingArguments.generate_factoids:
                        # apart from the intermediate conclusions, we also need the remaining factoids
                        # to generate the final conclusion
                        inputs = [base_input + " " + f"facts: {output};{';'.join(remaining_factoids)}"
                                  for output in model_outputs]
                    else:
                        # model outputs from previous step
                        inputs = [base_input + " " + f"facts: {output}" for output in model_outputs]
                model_inputs = _encode_inputs(tokenizer=self.tokenizer, inputs=inputs, targets=targets)
                loss += self.model(**model_inputs)[0]
                model_outputs = self.get_prediction(self.model, model_inputs)
                model_outputs = [output for output in model_outputs]
                # print(f"Model outputs at step {proof_step}: ", model_outputs)
                self.model.train()

        # check if loss is a tensor
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        avg_loss = loss / proof_steps
        labels = model_inputs.pop("labels")
        return avg_loss, model_outputs, labels

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prediction step
        :param model: model to train
        :param inputs: inputs to the model
        :param prediction_loss_only: whether to compute only the loss
        :param ignore_keys: keys to ignore
        :return: loss, logits, labels
        """
        # TASK1: Select factoids from the input
        model.eval()
        with torch.no_grad():
            # check inputs on device
            inputs = self._prepare_inputs(inputs)
            loss, outputs, labels = self.compute_prediction_loss(model, inputs, return_outputs=True)

        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        if isinstance(outputs, dict):
            logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
        else:
            logits = outputs[1:]

        if prediction_loss_only:
            return loss, None, None

        model.train()

        return loss, logits, labels


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

        self.train_set, self.val_set = TRAIN_DATA, VAL_DATA

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
            generate_base_input(self.train_args.prefix, ques, ans, hypothesis)
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

        def generate_target_input(factoid, feature):
            factoid_text = [feature[sent] for sent in factoid]
            return ";".join(factoid_text)

        base_inputs, inputs, targets = [], [], []

        model_outputs = []
        term_indices = []
        loss = 0
        remaining_factoids = []
        for proof_step in range(max_steps):
            hypotheses = examples["hypothesis"]
            meta_info = examples["meta"]
            explanation_chain = examples["explanation_chain"]
            predicted_factoids = []
            # at step 0, we generate the all factoids if generate_factoids is True
            # and filter the factoids that are required for the proof based on the explanation chain
            if proof_step == 0:
                # gold inputs from factoid generation step
                prev_inputs, prev_targets = self._preprocess_factoid_generation(examples)
                if self.train_args.generate_factoids:
                    logger.info("Generating factoids")
                    # generate factoids
                    model_inputs = _encode_inputs(self.tokenizer, prev_inputs, prev_targets)
                    loss += self.model(**model_inputs)[0]
                    predicted_factoids = self.predict(model_inputs)
                    prev_targets = predicted_factoids
                    predicted_factoids = list(
                        deepflatten([factoid.split(";") for factoid in predicted_factoids], depth=1))
                    # print("predicted_factoids: ", predicted_factoids)
                # set prev_inputs to base_inputs to be used in other steps
                base_inputs = prev_inputs

                # concatenate the previous inputs and targets
                # remove the trailing ; from the targets
                # NOTE: all factoids are used here
                prev_targets = [target.rstrip(";") for target in prev_targets]
                inputs = [base_input + " " + f"facts: {prev_target}"
                          for base_input, prev_target in zip(base_inputs, prev_targets)]
                # print("Inputs at step 0: ", inputs)

                # targets will be the gold factoids from the explanation chain
                sent_idx, int_idx = 0, 1
                selected_factoids = [
                    explanation[proof_step][sent_idx]
                    for explanation in explanation_chain
                ]

                targets = [generate_target_input(selected_factoids[i], meta_info[i]["triples"])
                           for i in range(len(selected_factoids))]

        # print("inputs: ", inputs)
        # print("targets: ", targets)
        # print("-" * 100)
        return inputs, targets



    def _preprocess_pipeline(self, examples):
        # print("Preprocessing pipeline")
        # task 1: Generate and select factoids
        # task 2 : Generate intermediate conclusion
        # task 3 : Generate final conclusion
        inputs, targets = self._preprocess_factoid_selection(examples)
        model_inputs = _encode_inputs(self.tokenizer, inputs, targets)
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

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.train_args.warmup_steps if self.train_args.warmup_steps > 0
            else math.ceil(num_training_steps * self.train_args.warmup_ratio)
        )
        return warmup_steps

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

            trainer = ExplanationTrainer(
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
