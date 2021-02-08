Gold Standard annotations for SMM4H-Spanish shared task. SMM4H 2021 accepted at NAACL (scheduled in Mexico City in June) https://2021.naacl.org/.


1. Introduction:
The entire corpus contains 10,000 annotated tweets. It has been split into training, validation and test (60-20-20). The current version contains the training and development set of the shared task with Gold Standard annotations.
In future versions of the dataset, test and background sets will be released.

For the subtask-1 (classification), annotations are distributed in a tab-separated file (TSV). The TSV format follows the format employed in SMM4H 2019 Task 2:
tweet_id	class
 
For the subtask-2 (Named Entity Recognition, profession detection), annotations are distributed in 2 formats: Brat standoff and TSV. See Brat webpage for more information about Brat standoff format (https://brat.nlplab.org/standoff.html). The TSV format follows the format employed in SMM4H 2019 Task 2:
tweet_id	begin	end	type	extraction

In addition, we provide a tokenized version of the dataset, for participant's convenience. It follows the BIO format (similar to CONLL). The files were generated with the brat_to_conll.py script (included), which employs the es_core_news_sm-2.3.1 Spacy model for tokenization.


2. Zip structure:
txt-files: folder with text files. One text file per tweet. One sub-directory per corpus split (train and valid)

subtask-1: annotation files of Named Entity Recognition subtask. It has one sub-directory per annotation format.

	brat: folder with annotations in Brat format. One sub-directory per corpus split (train and valid)
	TSV: folder with annotations in TSV. One file per corpus split (train and valid)
	BIO: folder with corpus in BIO tagging. One file per corpus split (train and valid)

subtask-2: annotation files of twitter classification subtask. One file per corpus split (train and valid).

3. Important shared task information:
System predictions must follow the TSV format. And systems will only be evaluated for the PROFESION and SITUACION_LABORAL predictions (despite the Gold Standard contains 2 extra entity classes). For more information about the evaluation scenario, see the Codalab link in the description, or the evaluation webpage (https://temu.bsc.es/smm4h-spanish/?p=3975).



For further information, please visit https://temu.bsc.es/smm4h-spanish/ or email us at encargo-pln-life@bsc.es

