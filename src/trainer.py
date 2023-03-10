"""
This file contains the trainer class for the generation of Chain of Thought explanations.
"""

import logging
import itertools
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from iteration_utilities import deepflatten, flatten
from typing import List, Tuple, Callable, Iterable

from datasets import load_metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import wandb

from src.dataloader import PreprocessDataset
from src.common.config import DataArguments, ModelArguments, TrainingArguments

logger = logging.getLogger(__name__)


class ExplanationTrainer(Trainer):
    """
    Trainer class for explanation generation
    Overrides the training and evaluation methods
    in huggingface's Trainer class
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs[0]
        print("Loss in compute loss: ", loss)
        if return_outputs:
            return loss, outputs
        return loss


class ExplanationTreeGenerator:
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

        # load data
        self._load_data()

    def _load_data(self):

        config = AutoConfig.from_pretrained(self.model_args.model_name_or_path, cache_dir=self.model_args.cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path, cache_dir=self.model_args.cache_dir, use_fast=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_args.model_name_or_path, config=config, cache_dir=self.model_args.cache_dir
        )
        self.model.to(self.device)
        dataloader = PreprocessDataset(
            data_path=self.data_args.data_path,
            seed=self.data_args.seed,
            shuffle=self.data_args.shuffle,
        )
        self.train_set, self.val_set = dataloader.preprocess_data()

    def _preprocess_factoid_generation(
        self,
        examples,
    ) -> Tuple[List[str], List[str]]:
        question = examples["question"]
        context = examples["context"]
        answer = examples["answer"]
        hypotheses = examples["hypothesis"]
        meta_info = examples["meta"]

        def generate_base_input(prefix, _question, _answer, _hypothesis):
            input = f"{prefix} question: {_question.lstrip()} answer: {_answer.lstrip()} hypothesis: {_hypothesis.lstrip()}"
            return input

        def generate_target_input(factoids):
            # replace none with empty string and joining with ;
            factoids = [factoid if factoid else "" for factoid in factoids]
            return ";".join(factoids)

        inputs = [
            generate_base_input(self.train_args.prefix, ques, ans, hypothesis)
            for ques, ans, hypothesis in zip(question, answer, hypotheses)
        ]
        print(inputs)
        targets = [
            # join the factoid with ; to make it a single string
            generate_target_input(list(meta["triples"].values()))
            for meta in meta_info
        ]
        print(targets)
        return inputs, targets

    def generate_inputs_and_targets(
            self,
            step,
            base_input,
            meta_info,
            exp_chain,
            hypothesis
    ):
        """
        Generate inputs and targets for the model
        """
        exp_step = exp_chain[step]
        print("exp_step: ", exp_step)
        inp_idx, out_idx = 0, 1
        inp = exp_step[inp_idx]
        out = exp_step[out_idx]

        print("inp: ", inp)
        print("meta_info: ", meta_info)

        inp_text = [meta_info["triples"][x] if "sent" in x
                    else meta_info["intermediate_conclusions"][x] for x in inp]
        print("inp_text: ", inp_text)
        inputs = [base_input + " " + f"facts: {';'.join(inp_text)}"]

        # outputs can have intermediate conclusions or the final conclusion (hypothesis)
        out_text = [meta_info["intermediate_conclusions"][x] if "int" in x else hypothesis for x in out]
        targets = [out_text for out_text in out_text]
        return inputs, targets

    def _preprocess_intermediate_conclusion_generation(
        self,
        examples,
    ) -> Tuple[List[str], List[str]]:

        proof_steps = examples["depth_of_proof"]
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
                print("----------")
                print("In proof step 0")
                # gold inputs from factoid generation step
                prev_inputs, prev_targets = self._preprocess_factoid_generation(examples)
                if self.train_args.generate_factoids:
                    logger.info("Generating factoids")
                    # generate factoids
                    model_inputs = self._encode_inputs(prev_inputs, prev_targets)
                    loss += self.model(**model_inputs)[0]
                    predicted_factoids = self.predict(model_inputs)
                    prev_targets = predicted_factoids
                    predicted_factoids = list(deepflatten([factoid.split(";") for factoid in predicted_factoids], depth=1))
                    print("predicted_factoids: ", predicted_factoids)
                # set prev_inputs to base_inputs to be used in other steps
                base_inputs = prev_inputs
                print("Prev. inputs at step 0: ", prev_inputs)
                # concatenate the previous inputs and targets
                # remove the trailing ; from the targets
                prev_targets = [target.rstrip(";") for target in prev_targets]
                inputs = [base_input + " " + f"factoids: {prev_target}"
                          for base_input, prev_target in zip(base_inputs, prev_targets)]
                print("Inputs at step 0: ", inputs)

                # targets will be the gold factoids from the explanation chain
                sent_idx, int_idx = 0, 1
                selected_factoids = [
                    explanation[proof_step][sent_idx]
                    for explanation in explanation_chain
                ]

                targets = [generate_target_input(selected_factoids[i], meta_info[i]["triples"])
                           for i in range(len(selected_factoids))]
                print("targets at step 0: ", targets)

                if not self.train_args.use_gold_inputs:
                    # select factoids using the model
                    model_inputs = self._encode_inputs(inputs=inputs, targets=targets)
                    loss += self.model(**model_inputs)[0]
                    model_outputs = self.predict(model_inputs)
                    print("Model outputs at step 0: ", model_outputs)
                    # remove the filtered factoids (model_outputs) from the predicted factoids list
                    remaining_factoids = [factoid for factoid in predicted_factoids
                                          if factoid not in model_outputs[0].split(";")]
                    print("Remaining factoids: ", remaining_factoids)

            else:
                print("----------")
                print(f"In proof step {proof_step}")
                print(examples["explanation_chain"])
                current_step = proof_step -1

                print("meta_info: ", meta_info)
                print("length of explanation chain: ", len(explanation_chain))

                for base_input, meta_info, hypothesis, exp_chain in zip(
                        base_inputs, meta_info, hypotheses, explanation_chain):
                    exp_step = exp_chain[current_step]
                    inp_idx, out_idx = 0, 1

                    # outputs can have intermediate conclusions or the final conclusion (hypothesis)
                    out = exp_step[out_idx]
                    out_text = [meta_info["intermediate_conclusions"][x] if "int" in x else hypothesis for x in out]
                    targets = [out_text for out_text in out_text]

                    if self.train_args.use_gold_inputs:
                        inp = exp_step[inp_idx]
                        inp_text = [meta_info["triples"][x] if "sent" in x
                                    else meta_info["intermediate_conclusions"][x] for x in inp]
                        inputs = [base_input + " " + f"facts: {';'.join(inp_text)}"]
                        print("inp_text: ", inp_text)
                    else:
                        # check if we are in the last step of hypothesis generation
                        # and if we need to use the generated factoids
                        if self.train_args.generate_factoids and proof_step == max_steps - 1:
                            # apart from the intermediate conclusions, we also need the remaining factoids
                            # to generate the final conclusion
                            inputs = [base_input + " " + f"facts: {output};{';'.join(remaining_factoids)}"
                                      for output in model_outputs]
                        else:
                            # model outputs from previous step
                            inputs = [base_input + " " + f"facts: {output}" for output in model_outputs]
                        model_inputs = self._encode_inputs(inputs=inputs, targets=targets)
                        loss += self.model(**model_inputs)[0]
                        model_outputs = self.predict(model_inputs)
                        model_outputs = [output for output in model_outputs]
                        print(f"Model outputs at step {proof_step}: ", model_outputs)

            # decrement proof steps by subtracting from proof steps
            proof_steps = [p-1 for p in proof_steps]
            # termination indices
            term_indices = [i for i, p in enumerate(proof_steps) if p < 0]
        print("term_indices: ", term_indices)
        print("loss: ", loss)
        # check if loss is a tensor
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        avg_loss = loss / max_steps

        print("avg_loss: ", avg_loss)
        print("Final inputs: ", inputs)
        print("Final targets: ", targets)
        return inputs, targets


    def _encode_inputs(self, inputs, targets):

        """
        Create input prompts for the model
        :param examples: sample from dataset
        :return: encoded data
        """
        print("Encoding inputs")
        print("Inputs: ", inputs)
        print("Targets: ", targets)
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.model_args.max_src_length,
            padding=self.model_args.padding,
            truncation=True,
            return_tensors="pt",
        )

        # Setup the tokenizer for targets
        if targets is not None:
            labels = self.tokenizer(
                text_target=targets,
                max_length=self.model_args.max_target_length,
                padding=self.model_args.padding,
                truncation=True,
                return_tensors="pt",
            )
            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss
            if self.model_args.padding == "max_length":
                labels["input_ids"] = labels["input_ids"].masked_fill(
                    labels["input_ids"] == self.tokenizer.pad_token_id, -100)

            model_inputs["labels"] = labels["input_ids"]
        # ensure tensors are on the right device
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        # print(model_inputs)
        return model_inputs

    def _preprocess_pipeline(self, examples):
        print("Preprocessing pipeline")
        # task 1: Generate factoids
        # task 2 : Generate intermediate conclusion
        # task 3 : Generate final conclusion
        inputs, targets = self._preprocess_intermediate_conclusion_generation(examples)
        model_inputs = self._encode_inputs(inputs, targets)
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
        print("Predicting...")
        # print("Model inputs: ", model_inputs)
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
            train_dataset = self.train_set.map(
                self._preprocess_pipeline,
                batched=True,
                batch_size=self.data_args.batch_size,
                desc="Running tokenizer on train dataset",
            )
            print("Train dataset: ", train_dataset)
            print("Train dataset length: ", len(train_dataset))
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
                logging_strategy='epoch',
                predict_with_generate=True,
                load_best_model_at_end=True,
                save_total_limit=2,
                disable_tqdm=False,
                report_to=["wandb"],
                remove_unused_columns=False,
                seed=self.train_args.seed,
                label_names=["labels"],  # it's important to log eval_loss
            )
            # print("Batch Size", args.train_batch_size)
            # print("Parallel Mode", args.parallel_mode)

            trainer = ExplanationTrainer(
                model=self.model,
                args=args,
                data_collator=self.data_collator,
                train_dataset=train_dataset if self.train_args.do_train else None,
                eval_dataset=eval_dataset if self.train_args.do_eval else None,
                # compute_metrics=self.compute_metrics,
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

            # if self.do_predict:
            #     logger.info("*** Predict ***")
            #     if CHECKPOINT is not None:
            #         model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
            #     else:
            #         model = self.model
            #     predictions = []
            #     for batch in range(0, len(eval_dataset), BATCH_SIZE):
            #         data = eval_dataset[batch: batch + BATCH_SIZE]
            #         prep_data = self.data_collator([data])
            #         model.eval()
            #         model.to(self.device)
            #         with torch.no_grad():
            #             # https://huggingface.co/blog/how-to-generate
            #             generated_ids = model.generate(
            #                 input_ids=prep_data["input_ids"][0].to(self.device),
            #                 attention_mask=prep_data["attention_mask"][0].to(self.device),
            #                 max_length=self.max_target_len,
            #                 use_cache=True,
            #                 num_beams=NUM_BEAMS,
            #                 length_penalty=0.6,
            #                 early_stopping=True,
            #             )
            #         outputs = self.tokenizer.batch_decode(
            #             generated_ids, skip_special_tokens=True
            #         )
            #
            #         if self.save_results:
            #             for prem, hyp, label, exp, dec_preds in zip(
            #                 data["premise"],
            #                 data["hypothesis"],
            #                 data["label"],
            #                 data["explanation"],
            #                 outputs,
            #             ):
            #                 predictions.append(
            #                     {
            #                         "premise": prem,
            #                         "hypothesis": hyp,
            #                         "label": label,
            #                         "explanation": exp,
            #                         "predicted_label": "Contradiction"
            #                         if dec_preds.startswith("contradiction")
            #                         else "Entailment",
            #                         "model_explanation": " ".join(
            #                             dec_preds.split(" ")[2:]
            #                         ).lstrip(),
            #                     }
            #                 )
            #             # print(predictions)
            #     with open(OUTPUT_DIR + "/outputs.json", "w") as f:
            #         f.write(json.dumps(predictions, indent=4))


if __name__ == "__main__":
    data_args = DataArguments()
    model_args = ModelArguments()
    training_args = TrainingArguments()

    wandb.init(
            project=TrainingArguments.project_name,
    )
    wandb.run.name = training_args.output_dir.split("/")[-1]
    trainer = ExplanationTreeGenerator(data_args, model_args, training_args)
    trainer.train()
