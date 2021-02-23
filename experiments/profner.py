#!/usr/bin/env python
# coding: utf-8
from biome.text import Dataset, Pipeline, TrainerConfiguration
from biome.text.hpo import TuneExperiment
import itertools
import os
from ray import tune


# In[ ]:


train_ds = Dataset.from_json("../preprocessing_inference/train_v1.json")
valid_ds = Dataset.from_json("../preprocessing_inference/valid_v1.json")


# In[ ]:


train_ds.rename_column_("tags_bio", "tags")
valid_ds.rename_column_("tags_bio", "tags")
train_ds.rename_column_("classification_label", "labels")
valid_ds.rename_column_("classification_label", "labels")


# In[ ]:


get_ipython().system('wget https://zenodo.org/record/4449930/files/cbow_cased.tar.gz')


# In[ ]:


get_ipython().system('tar -xzf cbow_cased.tar.gz')


# In[ ]:


get_ipython().system('head cased/covid_19_es_twitter_cbow_cased.vec')


# In[ ]:


profner = {
    "name": "profner",
    "features": {
        "word": {
            "embedding_dim": 300, 
            #"weights_file": "/content/cased/covid_19_es_twitter_cbow_cased.vec",
            "trainable": True,
        }
    },
    "encoder": {
        "type": "gru",
        "num_layers": 1,
        "bidirectional": True,
        "hidden_size": 128,
    },
    "head": {
        "type": "ProfNer",
        "classification_labels": train_ds.unique("labels"),
        "classification_pooler": {
            "type": "gru",
            "num_layers": 1,
            "bidirectional": True,
            "hidden_size": 64,
        },
        "ner_feedforward": {
            "activations": ["relu"],
            "dropout": [0],
            "hidden_dims": [128],
            "num_layers": 1,
        },
        "ner_tags": list(set(itertools.chain.from_iterable(train_ds["tags"]))),
        "ner_tags_encoding": "BIO",
        "dropout": 0.1,
    },
}


# In[ ]:


pipeline = Pipeline.from_config(profner)


# In[ ]:


pipeline.predict(["test", "this"])


# In[ ]:


trainer_config = dict(
    optimizer={
        "type": "adamw",
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(5e-3, 5e-2)
    },
    linear_decay=False,
    warmup_steps=tune.randint(0, 200),
    training_size=len(train_ds),
    batch_size=tune.choice([4, 8, 16]),
    num_epochs=tune.choice([3, 4, 5]),
    validation_metric="+ner/f1-measure-overall"
)


# In[ ]:


from ray.tune.suggest.hyperopt import HyperOptSearch


# In[ ]:


hyperopt = HyperOptSearch(metric="validation_ner/f1-measure-overall", mode="max", n_initial_points=2)


# In[ ]:


random_search = TuneExperiment(
    pipeline_config=profner,
    trainer_config=trainer_config,
    train_dataset=train_ds.select(range(32)),
    valid_dataset=valid_ds.select(range(32)),
    name="profner",
    num_samples=20,
    local_dir="tune_runs",
    resources_per_trial={"cpu": 5, "gpu": 0},
)


# In[ ]:


analysis = tune.run(
    random_search,
    config=random_search.config,
    scheduler=tune.schedulers.ASHAScheduler(),
    search_alg=hyperopt,
    metric="validation_ner/f1-measure-overall", 
    mode="max",
    progress_reporter=tune.JupyterNotebookReporter(overwrite=True)
)


# In[ ]:




