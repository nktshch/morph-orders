## Overview

More information will be added later

## Running code

### Python version

This project does not require certain package versions. However, it was only tested with python 3.12.12. We also suggest
using torch version with cuda available for better experience.

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

### Models

Trained models with be in `results` folder alongside logs and vocab files

### Evaluation

Will add later
