This folder contains everything to reproduce the data preprocessing and postprocessing for submitted predictions.

### preprocessing.ipynb

Produces the training/validation data sets (train_v*.json, valid_v*.json). 
Look at `versions.md` to check out the differences between v1, v2 and v3 of the data sets.

### inference.ipynb

Produces the submitted predictions

### transformer_tokenizer.ipynb, subword_pooling.ipynb

Playground to figure out some concepts (**not important for reproducing the results**).

### submissions

Contains the tsv and zip files of the various submissions