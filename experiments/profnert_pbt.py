from biome.text import Dataset, Pipeline, Trainer, VocabularyConfiguration
from pathlib import Path

from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
import os
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import PopulationBasedTraining

train_ds = Dataset.from_json("../preprocessing_inference/train_v2.json")
valid_ds = Dataset.from_json("../preprocessing_inference/valid_v2.json")

train_ds.rename_column_("tags_bio", "tags")
valid_ds.rename_column_("tags_bio", "tags")
train_ds.rename_column_("classification_label", "labels")
valid_ds.rename_column_("classification_label", "labels")

train_path = Path("./train_v2_data")
train_ds.save_to_disk(train_path.absolute())

valid_path = Path("./valid_v2_data")
valid_ds.save_to_disk(valid_path.absolute())


def trainable(config, checkpoint_dir: str = None, train_data_path: str = None, valid_data_path: str = None):
    train_dataset = Dataset.load_from_disk(train_data_path)
    valid_dataset = Dataset.load_from_disk(valid_data_path)

    tune_callback = TuneReportCheckpointCallback(
        metrics=[
            "valid_classification/accuracy",
            "valid_ner/f1-measure-overall",
            "valid_ner/f1-measure-PROFESION",
            "valid_ner/f1-measure-SITUACION_LABORAL",
        ]
    )

    # transformers_model: str = "prajjwal1/bert-tiny"
    transformers_model: str = "dccuchile/bert-base-spanish-wwm-cased"
    pipeline = Pipeline.from_config({
        "name": "profnert",
        "features": {
            "transformers": {
                "model_name": transformers_model,
                "trainable": True,
            }
        },
        "head": {
            "type": "ProfNerT",
            "classification_labels": ['1', '0'],
            "classification_pooler": {
                "type": "bert_pooler",
                "pretrained_model": transformers_model,
                "requires_grad": True,
                "dropout": 0.1,
            },
            "ner_tags": [
                'B-PROFESION',
                'I-PROFESION',
                'O',
                'B-SITUACION_LABORAL',
                'I-SITUACION_LABORAL'
            ],
            "ner_tags_encoding": "BIO",
            "transformers_model": transformers_model,
            "dropout": 0.1,
        },
    })
    pipeline.create_vocab(VocabularyConfiguration(datasets=[train_dataset]))

    trainer = Trainer(
        optimizer={
            "type": "adamw",
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
        },
        callbacks=[tune_callback],
        batch_size=8,
        max_epochs=5,
        gpus=1,
        progress_bar_refresh_rate=0,
        checkpoint_callback=False,
        limit_train_batches=0.05,
        limit_val_batches=0.05,
    )

    if checkpoint_dir:
        ckpt = pl_load(os.path.join(checkpoint_dir, "checkpoint"))
        pipeline.model.load_state_dict(ckpt["state_dict"])
        trainer.trainer.current_epoch = ckpt["epoch"]
        trainer.trainer.global_step = ckpt["global_step"]
    else:
        pass

    trainer.fit(
        pipeline,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        vocab_config=None,
    )


scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="valid_ner/f1-measure-overall",
    mode="max",
    perturbation_interval=1,
    hyperparam_mutations={
        "lr": tune.loguniform(1e-6, 1e-3),
        "weight_decay": tune.loguniform(1e-3, 1e-1),
    },
    synch=True,
)

analysis = tune.run(
    tune.with_parameters(
        trainable,
        train_data_path=str(train_path.absolute()),
        valid_data_path=str(valid_path.absolute()),
    ),
    config={
        "wandb": {
            "project": "profner_pbt",
            "log_config": True,
            "api_key": "505a6f09e4834d95e30906e7a7f006a3e686c448",
        },
    },
    num_samples=2,
    scheduler=scheduler,
    keep_checkpoints_num=1,
    checkpoint_score_attr="valid_ner/f1-measure-overall",
    progress_reporter=tune.CLIReporter(
        parameter_columns=["lr", "weight_decay"],
        metric_columns=["iter", "valid_classification/accuracy", "valid_ner/f1-measure-overall"],
    ),
    loggers=DEFAULT_LOGGERS + (WandbLogger,),
    resources_per_trial={"cpu": 2, "gpu": 1},
    local_dir="ray_tune_pbt",
)
