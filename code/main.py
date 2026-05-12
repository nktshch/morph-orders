"""Pipeline of the model."""

from data_preparation.vocab import get_vocab
from model.model import Model
from trainer import Trainer

import warnings
import argparse
from pathlib import Path
from datasets import load_dataset
import json
import torch
from torch import cuda
import numpy as np
import random


# controls what layers will be frozen in BERT-like encoders
FROZEN_PARAMS = [
    'embedding'
]


def parse_arguments():
    """Loads config from json file and adds command line arguments to it."""

    argp = argparse.ArgumentParser()
    argp.add_argument('config', help='json file containing configurations parameters')

    args = argp.parse_args()

    with open(args.config, 'r') as json_file:
        config = json.load(json_file)

    if config['loss'] not in ['xe', 'oaxe']:
        raise ValueError(f'Unknown loss: {config["loss"]}')

    config['word_LSTM_directions'] = 1 + int(config['word_LSTM_bidirectional'])
    config['char_LSTM_directions'] = 1 + int(config['char_LSTM_bidirectional'])
    config['teacher_forcing'] = False if config['loss'] == 'oaxe' else True
    # why???

    config['device'] = 'cuda' if cuda.is_available() else 'cpu'
    # config['device'] = 'cpu'

    return config


def get_folders(config, model_folder):
    config['model_folder'] = model_folder
    config['language_folder'] = 'data/' + config['language']
    config['train_files'] = [str(x) for x in Path(config['language_folder']).glob('*train*.parquet')]
    config['valid_files'] = [str(x) for x in Path(config['language_folder']).glob('*dev*.parquet')]
    config['test_files'] = [str(x) for x in Path(config['language_folder']).glob('*test*.parquet')]
    if len(config['train_files']) > 1 or len(config['valid_files']) > 1 or len(config['test_files']) > 1:
        warnings.warn('WARNING: More than one file for one of the datasets detected. Only the first will be used')

    config['vocab_file'] = f'{config["model_folder"]}/vocab.pickle'

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    return config


def freeze_params(model, verbose=1):
    """Freezes parameters in the encoder and prints summary."""

    frozen = 0
    total = 0
    for name, param in model.named_parameters():
        total += 1
        if 'encoder' in name:
            if model.conf['freeze'] == 'all':
                param.requires_grad = False
                frozen += 1
            elif model.conf['freeze'] == 'none':
                pass
            elif any(sub in name for sub in FROZEN_PARAMS):
                param.requires_grad = False
                frozen += 1
        if verbose > 0:
            print(f'\t{name}, requires_grad={param.requires_grad}')
    print(f'Model trainable params: {total - frozen} of {total}')


def multiple(grid=None):
    """Starts training for each language and each order specified in grid. If any other parameters specified in grid,
    they are treated as hyperparameters that need to be searched.

    Args:
        grid (dict): it should contain two fields: language and order. The rest are optional.
    """

    from itertools import product
    import math

    def run(seed):
        # run_number is used as random seed
        torch.backends.cudnn.deterministic = True
        torch.random.manual_seed(seed)
        cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = Model(conf, vocab, encoder_type=conf['encoder_type']).to(conf['device'])
        freeze_params(model, verbose=0)
        trainer = Trainer(conf, model, datasets['train'], datasets['valid'],
                          run_number=seed, subset_size=6, save_model=True).to(conf['device'])
        trainer.train()

        return trainer.best_accuracy, trainer.best_loss

    def iterate_param_grid(grid_dict):
        """For better iteration over fields in grid."""

        # Get parameter names and value lists
        param_names = list(grid_dict.keys())
        value_lists = list(grid_dict.values())

        # Generate all combinations
        for combination in product(*value_lists):
            # Create dictionary for this combination
            param_dict = dict(zip(param_names, combination))
            yield param_dict


    # this determines how many variants there are for each language and order
    grid_dims = [len(v) for k, v in grid.items() if (k != 'language' and k != 'order')]
    per_language_runs = math.prod(grid_dims)
    print(f'{per_language_runs} variant(s) for each language and order')

    search_results = []

    for i, params in enumerate(iterate_param_grid(grid)):
        conf = parse_arguments()
        if per_language_runs > 1 and conf['number_of_runs'] > 1:
            warnings.warn('You have hyperparameters specified in grid, but number of runs is greater than 1. This means'
                          'that model will be trained more than once with each value, which is likely not intended.')

        # params = {'language': <language>, 'order': <order>, ...}
        for k, v in params.items():
            conf[k] = v

        model_folder = f'results/{params["order"]}/{params["language"]}'
        get_folders(conf, model_folder)
        # for key in conf:
        #     print(f'{key} : {conf[key]}')

        print(f'Current language and order:\n\t{params["language"]}\n\t{params["order"]}\n')

        datasets = load_dataset('parquet', data_files={'train': conf['train_files'][0],
                                                       'valid': conf['valid_files'][0],
                                                       'test': conf['test_files'][0]})

        vocab = get_vocab(conf, datasets['train'], rewrite=True)
        vocab.create_embeddings(ft=None)

        # print('grammemes will be sorted in this order:')
        # for grammeme in list(vocab.sorting_order.keys()):
        #     print(f'{grammeme} - {vocab.vocab["grammeme-index"][grammeme]}')

        for run_number in range(conf['number_of_runs']):
            best_acc, best_lss = run(run_number)
            print(f'\n== Results ==')
            print(f'{params}\n{best_acc}, {best_lss}')
            search_results.append((params, best_acc, best_lss))

    # for now, all results are just dumped into one txt
    with open(f'results/all_results.txt', 'w+') as txt:
        previous_language = None
        previous_order = None
        for result in search_results:
            params, acc, lss = result
            current_language = params.get('language', None)
            current_order = params.get('order', None)

            if previous_language is not None and current_language != previous_language:
                txt.write('\n') # extra newline when language changes
            if previous_order is not None and current_order != previous_order:
                txt.write('\n') # another extra newline when order changes

            txt.write(f'{" ".join(list(params.values()))}\n')
            txt.write(f'\t{100 * acc:.2f}, {lss:.4f}\n')
            previous_language = current_language
            previous_order = current_order


if __name__ == "__main__":
    multiple(
        {
            'language': [
                'UD_Russian-GSD',
                # 'UD_Russian-SynTagRus',
            ],
            'order': [
                'standard',
                'reverse',
                'pos,reverse',
                'grammemes-down',
                'grammemes-up',
                'pos,grammemes-down',
                'pos,grammemes-up',
                'categories-down',
                'categories-up',
                'pos,categories-up'
            ]
        }
    )
