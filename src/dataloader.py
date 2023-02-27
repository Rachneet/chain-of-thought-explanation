import json
from datasets import load_dataset, Features, Value, concatenate_datasets
import numpy as np


class PreprocessDataset:
    """
    Preprocess the data
    """
    def __init__(self, data_path, seed, shuffle):
        self.data_path = data_path
        self.shuffle = shuffle
        self.seed = seed

    def _load_data(self):
        # load data in huggingface dataset
        train_data_file = {
            "train": f"{self.data_path}train.jsonl",
        }
        dev_data_file = {
            "dev": f"{self.data_path}dev.jsonl",
        }
        test_data_file = {
            "test": f"{self.data_path}test.jsonl",
        }
        train_data = load_dataset("json", data_files=train_data_file)["train"]
        dev_data = load_dataset("json", data_files=dev_data_file)["dev"]
        test_data = load_dataset("json", data_files=test_data_file)["test"]

        # print("Train data: ", train_data)
        # print("Val data: ", dev_data)
        # if seed is not None shuffle data else shuffle data
        if self.shuffle:
            train_data = train_data.shuffle(seed=self.seed)
            dev_data = dev_data.shuffle(seed=self.seed)
            test_data = test_data.shuffle(seed=self.seed)
        else:
            train_data = train_data.shuffle()
            dev_data = dev_data.shuffle()
            test_data = test_data.shuffle()

        return train_data, dev_data, test_data

    def _process_proof_steps(self, example):
        # print("Example proof: ", example["proof"])
        proofs = list(filter(str.strip, example["proof"].split(';')))
        assert len(proofs) == example["length_of_proof"]
        explanation_chain: dict = {}
        for i in range(len(proofs)):
            # print("Proof ", i, ": ", proofs[i])
            proof = list(map(str.strip, proofs[i].split("->")[0].split("&")))
            int = list(map(str.strip, proofs[i].split("->")[1].split(":")))
            # example["proof" + str(i)] = proof
            # example["int" + str(i)] = [int[0]]
            explanation_chain[f"proof_step_{i}"] = [proof, [int[0]]]
            # print("explanation chain: ", explanation_chain)
        example["explanation_chain"] = list(explanation_chain.values())
        return example

    def preprocess_data(self):
        # preprocess data
        train_data, dev_data, test_data = self._load_data()
        train_data = train_data.map(self._process_proof_steps)
        dev_data = dev_data.map(self._process_proof_steps)
        # print("Train data: ", train_data[0])
        return train_data, dev_data


if __name__ == '__main__':
    data_path = './data/entailment_trees_emnlp2021_data_v3/dataset/task_1/'
    batch_size = 32
    num_workers = 4
    shuffle = True
    drop_last = True
    data_loader = PreprocessDataset(data_path, seed=42, shuffle=True)
    train_data, dev_data = data_loader.preprocess_data()

    proof_length = []
    for sample in train_data:
        length_of_proof = sample["length_of_proof"]
        proof_length.append(length_of_proof)
    from collections import Counter
    print(Counter(proof_length))
