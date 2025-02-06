import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from typing import Literal, Sequence

from omegaconf import OmegaConf
from transformers import Trainer, TrainingArguments

from musicbert_hf.checkpoints import (
    load_musicbert_multitask_token_classifier_from_fairseq_checkpoint,
    load_musicbert_token_classifier_from_fairseq_checkpoint,
)
from musicbert_hf.data import HDF5Dataset, collate_for_musicbert_fn
from musicbert_hf.metrics import compute_metrics, compute_metrics_multitask
from musicbert_hf.musicbert_class import freeze_layers


@dataclass
class Config:
    # data_dir should have train, valid, and test subdirectories
    data_dir: str
    output_dir_base: str
    checkpoint_path: str
    targets: str | list[str]
    conditioning: str | list[str] | None = None
    log_dir: str = os.path.expanduser("~/tmp/musicbert_hf_logs")
    # We will always load from a checkpoint so we don't need to specify architecture
    # architecture: Literal["base", "tiny"] = "base"
    num_epochs: int = 0
    batch_size: int = 4
    learning_rate: float = 2.5e-4
    warmup_steps: int = 0
    max_steps: int = -1
    wandb_project: str | None = None
    # If None, freeze all layers; if int, freeze all layers up to and including
    #   the specified layer; if sequence of ints, freeze the specified layers
    freeze_layers: int | Sequence[int] | None = None

    def __post_init__(self):
        assert self.num_epochs is not None or self.max_steps is not None, (
            "Either num_epochs or max_steps must be provided"
        )
        self._job_id = os.environ.get("SLURM_JOB_ID", None)
        if self._job_id is None:
            # Use the current time as the job ID if not running on the cluster
            self._job_id = str(time.time())

        if isinstance(self.targets, str):
            self.targets = [self.targets]

        if isinstance(self.conditioning, str):
            self.conditioning = [self.conditioning]
        elif self.conditioning is None:
            self.conditioning = []

    @property
    def train_dir(self) -> str:
        return os.path.join(self.data_dir, "train")

    @property
    def valid_dir(self) -> str:
        return os.path.join(self.data_dir, "valid")

    @property
    def test_dir(self) -> str:
        return os.path.join(self.data_dir, "test")

    @property
    def output_dir(self) -> str:
        return os.path.join(self.output_dir_base, self._job_id)

    def target_paths(self, split: Literal["train", "valid", "test"]) -> list[str]:
        return [
            os.path.join(self.data_dir, split, f"{target}.h5")
            for target in self.targets
        ]

    def conditioning_paths(self, split: Literal["train", "valid", "test"]) -> list[str]:
        return [
            os.path.join(self.data_dir, split, f"{cond}.h5")
            for cond in self.conditioning
        ]

    @property
    def multitask(self) -> bool:
        return len(self.targets) > 1


def get_dataset(config, split):
    data_dir = getattr(config, f"{split}_dir")
    train_dataset = HDF5Dataset(
        os.path.join(data_dir, "events.h5"),
        config.target_paths(split),
    )
    return train_dataset


def get_config_and_training_kwargs():
    conf = OmegaConf.from_cli(sys.argv[1:])
    config_fields = set(Config.__dataclass_fields__.keys())
    config_kwargs = {k: v for k, v in conf.items() if k in config_fields}
    training_kwargs = {k: v for k, v in conf.items() if k not in config_fields}
    config = Config(**config_kwargs)  # type:ignore
    return config, training_kwargs


if __name__ == "__main__":
    config, training_kwargs = get_config_and_training_kwargs()

    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project
    else:
        os.environ.pop("WANDB_PROJECT", None)
        # os.environ["WANDB_DISABLED"] = "true"

        # Uncomment to turn on model checkpointing (up to 100Gb)
        # os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    train_dataset = get_dataset(config, "train")
    valid_dataset = get_dataset(config, "valid")

    if config.checkpoint_path:
        if config.multitask:
            if config.conditioning:
                raise NotImplementedError(
                    "Conditioning is not yet implemented for multitask training"
                )
            else:
                model = (
                    load_musicbert_multitask_token_classifier_from_fairseq_checkpoint(
                        config.checkpoint_path,
                        checkpoint_type="musicbert",
                        num_labels=train_dataset.vocab_sizes,
                    )
                )
        else:
            if config.conditioning:
                raise NotImplementedError(
                    "Conditioning is not yet implemented for single-task training"
                )
            else:
                model = load_musicbert_token_classifier_from_fairseq_checkpoint(
                    config.checkpoint_path,
                    checkpoint_type="musicbert",
                    num_labels=train_dataset.vocab_sizes[0],
                )
    else:
        raise ValueError("checkpoint_path must be provided")

    freeze_layers(model, config.freeze_layers)

    training_kwargs = (
        dict(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            warmup_steps=config.warmup_steps,
            logging_dir=config.log_dir,
            max_steps=config.max_steps,
            push_to_hub=False,
            eval_on_start=True,
            eval_strategy="steps",
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=2,
            load_best_model_at_end=True,
        )
        | training_kwargs
    )
    if config.wandb_project:
        training_kwargs["report_to"] = "wandb"
    else:
        training_kwargs["report_to"] = None

    training_args = TrainingArguments(**training_kwargs)

    if config.multitask:
        compute_metrics_fn = partial(
            compute_metrics_multitask, task_names=config.targets
        )
    else:
        compute_metrics_fn = compute_metrics

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=partial(collate_for_musicbert_fn, multitask=config.multitask),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_loss_func=partial(model.compute_loss),
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()
