from dataclasses import dataclass, field, Field
from typing import List, Optional
import copy

BASE_PATH="/storage/ukp/work/sachdeva/research_projects/chain-of-thought"

@dataclass
class DataArguments:
    """
    Arguments for the data
    """
    data_path: str = field(
        default=BASE_PATH + "/src/data/entailment_trees_emnlp2021_data_v3/dataset/task_1/",
        metadata={
            "help": "Path to the data"
        }
    )
    seed: int = field(
        default=42,
        metadata={
            "help": "Random seed"
        }
    )
    shuffle: bool = field(
        default=True,
        metadata={
            "help": "Shuffle the data"
        }
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
    default="t5-small",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    max_src_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    padding: Optional[str] = field(
        default="max_length",
        metadata={
            "help": "Padding strategy"
        }
    )

@dataclass
class TrainingArguments:
    seed: int = field(
        default=42, metadata={"help": "Random seed for data split"}
    )
    batch_size: int = field(
        default=8, metadata={"help": "Batch size for training"}
    )
    do_train: bool = field(
        default=True, metadata={"help": "Whether to run training."}
    )
    do_eval: bool = field(
        default=True, metadata={"help": "Whether to run eval on the dev set."}
    )
    do_predict: bool = field(
        default=True, metadata={"help": "Whether to run predictions on the test set."}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={
            "help": "learning rate for training"
        },
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "Learning rate scheduler to use. One of ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']"
        },
    )
    output_dir: str = field(
        default=BASE_PATH+"/t5-small-cot",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Overwrite the content of the output directory"},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to a checkpoint file to resume training from."
        },
    )
    epochs: int = field(
        default=1,
        metadata={
            "help": "Number of epochs to train the model"
        },
    )
    weight_decay: float = field(
        default=0.0,
        metadata={
            "help": "Weight decay for training"
        },
    )
    warmup_steps: int = field(
        default=0,
        metadata={
            "help": "Number of warmup steps for learning rate scheduler."
        },
    )
    max_train_samples: Optional[int] = field(
        default=2,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    dropout: float = field(
        default=0.1,
        metadata={
            "help": "Dropout probability"
        },
    )
    num_beams: int = field(
        default=4,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "``model.predict`` and ``model.score`` methods."
        },
    )
    project_name: str = field(
        default="chain-of-thought",
        metadata={
            "help": "Name of the wandb project"
        },
    )
    prefix: str = field(
        default="Explanation Tree:",
        metadata={
            "help": "Prefix to use for generative models"
        },
    )
    generate_factoids: bool = field(
        default=True,
        metadata={
            "help": "Whether to generate factoids or not"
        },
    )
    use_gold_inputs: bool = field(
        default=True,
        metadata={
            "help": "Whether to use gold inputs or not"
        },
    )