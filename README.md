## Overview

More information will be added later

## Running code

### Python version

The easiest way to set up environment is to run 

```bash
conda env create -f environment.yml
```
which creates `mo-env` in conda. However, after that you have to additionally install `torch`. We recommend using version 2.11.0 with CUDA
which can be installed with this command:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### Data

This project uses UD treebanks that you can download from https://universaldependencies.org/download.html

They use CoNLL-U annotation format that includes other fields besides morphological features, such as lemmas and dependency relations.
However, we do not need them, and for the ease of usage convert CoNLL-U data to parquet format. This can be done
automatically with `conllu2parquet.py` script located in `code/data_preparation`. It expects datasets to be located in
their respective folder in `data`, i.e `data/UD_Russian-GSD`. All you need to do is run

```bash
python code/data_preparation/conllu2parquet.py
```

### Training

To run the code, you have to specify which config file to use. We provide default.json as the main option with tuned
hyperparameters. Run the code with this command:

```bash
python code/main.py configs/default.json
```

For now, the corpora and orders that you wish to test are specified in the `main.py` file.

### Models

Trained models with be in `results` folder alongside logs and vocab files

### Evaluation

To get metrics for a certain model, you can run

```bash
python code/evaluate.py [language] [order] --seed [seed]
```


