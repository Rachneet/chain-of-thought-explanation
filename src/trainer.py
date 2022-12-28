"""
Generation of explanation trees
"""

import logging
import json
import numpy as np
from typing import List, Tuple, Callable, Iterable

from datasets import load_metric
import torch
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


class ExplanationTreeTrainer:
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
        dataloader = PreprocessDataset(
            data_path=self.data_args.data_path,
            seed=self.data_args.seed,
            shuffle=self.data_args.shuffle,
        )
        self.train_set, self.val_set = dataloader.preprocess_data()

    def generate_base_input(self, prefix, _question, _answer, _hypothesis):
        input = f"{prefix} question: {_question.lstrip()} answer: {_answer.lstrip()} hypothesis: {_hypothesis.lstrip()}"
        return input

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
            # replace none with empty string and joing with ;
            factoids = [factoid if factoid else "" for factoid in factoids]
            return ";".join(factoids)

        inputs = [
            self.generate_base_input(self.train_args.prefix, ques, ans, hypothesis)
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

    def _preprocess_intermediate_conclusion_generation(
        self,
        examples,
    ) -> Tuple[List[str], List[str]]:
        question = examples["question"]
        context = examples["context"]
        answer = examples["answer"]
        hypotheses = examples["hypothesis"]
        meta_info = examples["meta"]
        explanation_chain = examples["explanation_chain"]

        proof_steps = examples["depth_of_proof"]
        print(proof_steps)
        max_steps = max(proof_steps) + 1

        def generate_target_input(factoid, feature):
            factoid_text = [feature[sent] for sent in factoid]
            return ";".join(factoid_text)

        base_inputs, inputs, targets = [], [], []
        term_indices = []
        for proof_step in range(max_steps):
            if proof_step == 0:
                print("----------")
                print("In proof step 0")
                print(examples["explanation_chain"])
                # inputs from factoid generation step
                prev_inputs, prev_targets = self._preprocess_factoid_generation(examples)
                # set prev_inputs to base_inputs to be used in other steps
                base_inputs = prev_inputs
                print("Prev. inputs at step 0: ", prev_inputs)
                if self.train_args.use_gold_inputs:
                    # concatenate the previous inputs and targets
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
                    # print(selected_factoids_text)
                else:
                    # TODO: use factoids generated by the model
                    pass

            else:
                print("----------")
                print(f"In proof step {proof_step}")
                print(examples["explanation_chain"])
                current_step = proof_step -1
                # use inputs and targets from the explanation chain
                def generate_inputs_and_targets(base_input, meta_info, exp_chain, hypothesis):
                    exp_step = exp_chain[current_step]
                    print("exp_step: ", exp_step)
                    inp_idx, out_idx = 0, 1
                    inp = exp_step[inp_idx]
                    out = exp_step[out_idx]
                    # input can have sentences or the intermediate conclusion
                    inp_text = [meta_info["triples"][x] if "sent" in x
                                else meta_info["intermediate_conclusions"][x] for x in inp]
                    print("inp_text: ", inp_text)
                    # outputs can have intermediate conclusions or the final conclusion (hypothesis)
                    out_text = [meta_info["intermediate_conclusions"][x] if "int" in x else hypothesis for x in out]
                    inputs = [base_input + " " + f"facts: {';'.join(inp_text)}"][0]
                    targets = [out_text for out_text in out_text][0]
                    return inputs, targets

                # inputs from previous step
                processed_data = [generate_inputs_and_targets(base_input, meta_info, exp_chain, hypotheses)
                          if idx not in term_indices else (inputs[idx], targets[idx])
                          for idx, (base_input, meta_info, exp_chain, hypotheses)
                          in enumerate(zip(base_inputs, meta_info, explanation_chain, hypotheses))]
                print(f"Data at step {proof_step}: ", processed_data)
                inputs = [inp for inp, _ in processed_data]
                targets = [out for _, out in processed_data]
                print("inputs at step {}: ".format(proof_step), inputs)
                print("targets at step {}: ".format(proof_step), targets)

            # decrement proof steps by subtracting from proof steps
            proof_steps = [p-1 for p in proof_steps]
            # termination indices
            term_indices = [i for i, p in enumerate(proof_steps) if p < 0]
        print("term_indices: ", term_indices)
        print("New proof steps", proof_steps)

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
            inputs, max_length=self.model_args.max_src_length, padding=self.model_args.padding, truncation=True
        )
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.model_args.max_target_length,
                padding=self.model_args.padding,
                truncation=True,
            )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.model_args.padding == "max_length":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        print(model_inputs)
        return model_inputs

    def _preprocess_pipeline(self, examples):
        # task 1: Generate factoids
        if self.train_args.generate_factoids:
            inputs, targets = self._preprocess_factoid_generation(examples)
            model_inputs = self._encode_inputs(inputs, targets)

        # task 2 : Generate intermediate conclusion
        inputs, targets = self._preprocess_intermediate_conclusion_generation(examples)
        model_inputs = self._encode_inputs(inputs, targets)
        # task 3 : Generate final conclusion

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

    def compute_metrics(self, eval_preds):

        label_2_id = {"contradiction": 0, "entailment": 1}
        pred_ids = eval_preds.predictions
        label_ids = eval_preds.label_ids

        pred_str = self.tokenizer.batch_decode(
            torch.tensor(pred_ids, device=self.model.device), skip_special_tokens=True)
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(
            torch.tensor(label_ids, device=self.model.device), skip_special_tokens=True)

        labels = [label_2_id.get(label.split()[0]) for label in label_str]
        preds = [label_2_id.get(pred.split()[0], 2) for pred in pred_str]

        metrics = dict()

        accuracy_metric = load_metric('accuracy')
        precision_metric = load_metric('precision')
        recall_metric = load_metric('recall', zero_division=0)
        f1_metric = load_metric('f1')

        metrics.update(accuracy_metric.compute(predictions=preds, references=labels))
        metrics.update(precision_metric.compute(predictions=preds, references=labels, average='weighted'))
        metrics.update(recall_metric.compute(predictions=preds, references=labels, average='weighted'))
        metrics.update(f1_metric.compute(predictions=preds, references=labels, average='weighted'))

        # log metrics to wandb
        wandb.log({
            "accuracy": metrics["accuracy"],
        })

        return metrics

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
                batch_size=self.train_args.batch_size,
                desc="Running tokenizer on train dataset",
            )
            # if self.do_eval or self.do_predict:
            #     if self.max_val_samples is not None:
            #         # We will select sample from whole data if argument is specified
            #         max_val_samples = min(len(self.val_set), self.max_val_samples)
            #         self.val_set = self.val_set.select(range(max_val_samples))
            #     eval_dataset = self.val_set.map(
            #         self._prepare_features,
            #         batched=True,
            #         desc="Running tokenizer on validation dataset",
            #     )
            #
            # args = Seq2SeqTrainingArguments(
            #     output_dir=OUTPUT_DIR,
            #     overwrite_output_dir=False,
            #     do_train=self.do_train,
            #     do_eval=self.do_eval,
            #     per_device_train_batch_size=config.batch_size,
            #     per_device_eval_batch_size=BATCH_SIZE,
            #     learning_rate=config.learning_rate,
            #     num_train_epochs=config.epochs,
            #     weight_decay=config.weight_decay,
            #     lr_scheduler_type=SCHEDULER,
            #     save_strategy='epoch',
            #     evaluation_strategy='epoch',
            #     logging_strategy='epoch',
            #     predict_with_generate=True,
            #     load_best_model_at_end=True,
            #     save_total_limit=2,
            #     # run_name=WANDB_RUN_NAME,
            #     disable_tqdm=False,
            #     report_to=["wandb"],
            #     remove_unused_columns=False,
            #     fp16=FP16,
            #     seed=SEED,
            #     label_names=["labels"],  # it's important to log eval_loss
            # )
            # # print("Batch Size", args.train_batch_size)
            # # print("Parallel Mode", args.parallel_mode)
            #
            # trainer = Seq2SeqTrainer(
            #     model=self.model,
            #     args=args,
            #     data_collator=self.data_collator,
            #     train_dataset=train_dataset if self.do_train else None,
            #     eval_dataset=eval_dataset if self.do_eval else None,
            #     compute_metrics=self.compute_metrics,
            # )
            # try:
            #     if self.do_train:
            #         checkpoint = None
            #         if RESUME_TRAINING is not None:
            #             checkpoint = RESUME_TRAINING
            #         trainer.train(resume_from_checkpoint=checkpoint)
            #         trainer.save_model()
            # except KeyboardInterrupt:
            #     trainer.save_model("interrupted-fig-lang")
            #
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
    trainer = ExplanationTreeTrainer(data_args, model_args, training_args)
    trainer.train()
