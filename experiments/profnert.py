#!/usr/bin/env python
# coding: utf-8
import itertools
from biome.text import Dataset, Pipeline, TrainerConfiguration, helpers
from biome.text.hpo import TuneExperiment, TuneMetricsLogger
from biome.text.loggers import is_wandb_installed_and_logged_in, WandBLogger
from ray import tune

# In[ ]:


train_ds = Dataset.from_json("/home/ec2-user/biome/profner/preprocessing_inference/train_v2.json")
valid_ds = Dataset.from_json("/home/ec2-user/biome/profner/preprocessing_inference/valid_v2.json")
#train_ds = Dataset.from_json(
#    "/home/david/recognai//projects/ProfNER/profner/preprocessing_inference/train_v2.json"
#)
#valid_ds = Dataset.from_json(
#    "/home/david/recognai//projects/ProfNER/profner/preprocessing_inference/valid_v2.json"
#)


# In[ ]:


train_ds.rename_column_("tags_bio", "tags")
valid_ds.rename_column_("tags_bio", "tags")
train_ds.rename_column_("classification_label", "labels")
valid_ds.rename_column_("classification_label", "labels")


# In[ ]:
transformers_model: str = "dccuchile/bert-base-spanish-wwm-cased"
# transformers_model: str = "prajjwal1/bert-tiny"

profnert = {
    "name": "profnert",
    "features": {
        "transformers": {
            "model_name": transformers_model,
            "trainable": True,
        }
    },
    "head": {
        "type": "ProfNerT",
        "classification_labels": train_ds.unique("labels"),
        "classification_pooler": {
            "type": "bert_pooler",
            "pretrained_model": transformers_model,
            "requires_grad": True,
            "dropout": 0.1,
        },
        "ner_tags": list(set(itertools.chain.from_iterable(train_ds["tags"]))),
        "ner_tags_encoding": "BIO",
        "transformers_model": transformers_model,
        "dropout": 0.1,
    },
}

# In[ ]:

trainer_config = dict(
    optimizer={
        "type": "adamw",
        "lr": tune.loguniform(5e-6, 1e-4),
        "weight_decay": tune.loguniform(1e-3, 1e-1),
    },
    linear_decay=True,
    warmup_steps=tune.randint(0, 200),
    # BayesOpt does not support integers, these are a workaround
    # warmup_steps=tune.uniform(0, 200),
    batch_size=tune.choice([8, 16]),
    # batch_size=tune.uniform(1, 4),
    num_epochs=tune.choice([3, 4, 5]),
    # num_epochs=tune.uniform(3, 6),
    validation_metric="+valid_ner/f1-measure-overall",
    num_serialized_models_to_keep=0,
)

from ray.tune.suggest.hyperopt import HyperOptSearch
search_alg = HyperOptSearch(metric="validation_valid_ner/f1-measure-overall", mode="max")

# does not support integers
#from ray.tune.suggest.bayesopt import BayesOptSearch

#search_alg = BayesOptSearch(
#    metric="validation_valid_ner/f1-measure-overall", mode="max"
#)

# ray2.0.0 does not use the gpu ...
# from ray.tune.suggest.hebo import HEBOSearch
# search_alg = HEBOSearch(metric="validation_ner/f1-measure-overall", mode="max")


# In[ ]:

def default_trainable(config, reporter):
    """A default trainable function used by `tune.run`

    It performs the most straight forward training loop with the provided `config`:
    - Create the pipeline (optionally with a provided vocab)
    - Set up a MLFlow and WandB logger
    - Set up a TuneMetrics logger that reports all metrics back to ray tune after each epoch
    - Create the vocab if necessary
    - Execute the training
    """
    # treat discrete variables
    warmup_steps = config["trainer_config"]["warmup_steps"]
    config["trainer_config"]["warmup_steps"] = int(warmup_steps)

    batch_size = config["trainer_config"]["batch_size"]
    if int(batch_size) == 1:
        config["trainer_config"]["batch_size"] = 8
    elif int(batch_size) == 2:
        config["trainer_config"]["batch_size"] = 16
    elif int(batch_size) == 3:
        config["trainer_config"]["batch_size"] = 32

    num_epochs = config["trainer_config"]["num_epochs"]
    if int(num_epochs) == 3:
        config["trainer_config"]["num_epochs"] = 3
    elif int(num_epochs) == 4:
        config["trainer_config"]["num_epochs"] = 4
    elif int(num_epochs) == 5:
        config["trainer_config"]["num_epochs"] = 5

    pipeline = Pipeline.from_config(
        config["pipeline_config"], vocab_path=config["vocab_path"]
    )

    trainer_config = TrainerConfiguration(
        **helpers.sanitize_for_params(config["trainer_config"])
    )

    train_ds = Dataset.load_from_disk(config["train_dataset_path"])
    valid_ds = Dataset.load_from_disk(config["valid_dataset_path"])

    train_loggers = []
    train_loggers += [TuneMetricsLogger()]
    if is_wandb_installed_and_logged_in():
        train_loggers = [WandBLogger(project_name=config["name"])] + train_loggers

    pipeline.train(
        output="training",
        training=train_ds,
        validation=valid_ds,
        trainer=trainer_config,
        loggers=train_loggers,
        vocab_config=None if config["vocab_path"] else "default",
    )


hpo_experiment = TuneExperiment(
    pipeline_config=profnert,
    trainer_config=trainer_config,
    train_dataset=train_ds,
    valid_dataset=valid_ds,
    name="profner_transformers",
    num_samples=100,
    local_dir="tune_runs",
    resources_per_trial={"cpu": 2, "gpu": 1},
    # trainable=default_trainable,
)


# In[ ]:


analysis = tune.run(
    hpo_experiment,
    config=hpo_experiment.config,
    scheduler=tune.schedulers.ASHAScheduler(),
    search_alg=search_alg,
    metric="validation_valid_ner/f1-measure-overall",
    mode="max",
    progress_reporter=tune.CLIReporter(
        metric_columns=[
            "best_validation_valid_ner/f1-measure-overall",
            "best_validation_valid_classification/accuracy",
        ],
        parameter_columns=["trainer.optimizer.lr"],
    ),
    # progress_reporter=tune.JupyterNotebookReporter(overwrite=True)
    verbose=1,
)
