"""Contains methods to compare models."""

# import conll18_ud_eval
from trainer import masked_select
from model.model import Model
from data_preparation.vocab import Vocab

import re
import pickle
import argparse
import torch
from tqdm import tqdm
from time import time
from pathlib import Path
from collections import defaultdict, OrderedDict, Counter
from torch import cuda
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
import pandas as pd


DEVICE = 'cuda' if cuda.is_available() else 'cpu'
NUMBER_OF_RUNS = 5


def parse_arguments():
    argp = argparse.ArgumentParser()
    argp.add_argument('language', help='Language')
    argp.add_argument('order', help='Order of prediction')
    argp.add_argument('--seed', help='Random seed. If not given, every seed will be evaluated')

    return vars(argp.parse_args())


def load_all(model_file, vocab_file):
    print(f'Loading from {model_file}\n')
    # import pathlib
    # temp = pathlib.PosixPath
    # pathlib.PosixPath = pathlib.WindowsPath

    conf, state_dict = torch.load(model_file)
    with open(vocab_file, 'rb') as vf:
        vocab = pickle.load(vf)

    vocab.embeddings = state_dict['encoder.word_embeddings.weight'].detach().cpu().numpy()
    model = Model(conf, vocab, conf['encoder_type'])
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # pathlib.PosixPath = temp
    print(f'Loaded model, conf, vocab')
    return conf, vocab, model


class Evaluator:
    def __init__(self, parameters):
        self.language = parameters['language']
        self.order = parameters['order']
        self.seed = parameters['seed']

        # load conf, vocab, model
        if self.seed:
            model_file = f'results/{self.order}/{self.language}/model_{self.seed}.pt'
        else:
            # TODO: solve this
            raise ValueError('Please specify seed')
        vocab_file = f'results/{self.order}/{self.language}/vocab.pickle'
        self.conf, self.vocab, self.model = load_all(model_file, vocab_file)

        # init loader
        parquet_input = str(next(Path(f'data/{self.language}').glob('*test*.parquet')))
        self.input_data = load_dataset('parquet', data_files={'test': parquet_input})['test']
        self.loader = DataLoader(
            self.input_data,
            batch_size=self.conf['sentence_eval_batch_size'],
            collate_fn=lambda batch: self.vocab.collate_fn(batch, train_mode=False))

        # output_data holds annotated dataset
        self.output_data = None

        # metrics
        self.accuracy = None
        self.fscores = None


    def fill(self):
        grammemes = []
        progress_bar = tqdm(self.loader, total=len(self.loader), colour='#0011dd')
        for words_batch, chars_batch, _ in progress_bar:
            # here we don't need labels - calculate metrics later
            words_batch = torch.tensor(words_batch, dtype=torch.long).to(self.conf['device'])
            words_batch = words_batch.permute(1, 0)
            chars_batch = torch.tensor(chars_batch, dtype=torch.long).to(self.conf['device'])
            chars_batch = chars_batch.view(-1, chars_batch.shape[2]).permute(1, 0)

            predictions, _ = self.model(words_batch, chars_batch, None)
            current_batch_size = words_batch.shape[1]
            predictions = predictions.permute(1, 0)
            predictions = masked_select(predictions, self.vocab.vocab['grammeme-index'][self.conf['EOS']])
            predictions = predictions.view(current_batch_size, -1, predictions.shape[1])

            grammemes_batch = self.predictions_to_grammemes(predictions)
            grammemes.extend(grammemes_batch)

        return grammemes


    def predictions_to_grammemes(self, predictions):
        """Turns indices of predictions produced by decoder into actual grammemes (strings).

        Args:
            predictions (torch.Tensor): 2D Tensor containing indices.

        Returns:
            list: List of lists of predicions.
        """

        grammemes_batch = []
        for item in predictions:
            sentence = []
            for tag_indices in item:
                tag = []
                for grammeme_index in tag_indices:
                    if self.vocab.vocab['index-grammeme'][grammeme_index.item()] == self.conf['PAD']:
                        break
                    tag += [self.vocab.vocab['index-grammeme'][grammeme_index.item()]]
                sentence += [tag]
            grammemes_batch += [sentence]
        return grammemes_batch


    def create_parquet(self):
        grammemes = self.fill()

        output_data = self.input_data.remove_columns('tags')
        output_data = output_data.add_column('tags', grammemes)
        self.output_data = output_data
        return output_data


    def calculate_accuracy(self):
        input_tags = [t for token in self.input_data['tags'] for t in token]
        output_tags = [t for token in self.output_data['tags'] for t in token]

        correct = sum(1 for gt, pred in zip(input_tags, output_tags)
                      if len(gt) == len(pred) and all(g == p for g, p in zip(gt, pred)))

        self.accuracy = (correct / len(output_tags)) * 100
        print(f'Accuracy:\t\t\t\t\t{self.accuracy:.2f}%')


    def calculate_fscore(self):
        input_tags = [t for token in self.input_data['tags'] for t in token]
        output_tags = [t for token in self.output_data['tags'] for t in token]

        all_grammemes = set()
        for gt in input_tags:
            all_grammemes.update(gt)
        for pred in output_tags:
            all_grammemes.update(pred)

        # counters for each grammeme
        stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

        for gt, pred in zip(input_tags, output_tags):
            gt_set = set(gt)
            pred_set = set(pred)

            # update counters for each grammeme
            for grammeme in all_grammemes:
                in_gt = grammeme in gt_set # if present in token from input
                in_pred = grammeme in pred_set # if present in token from output

                if in_gt and in_pred: # true positive
                    stats[grammeme]['tp'] += 1
                elif in_pred and not in_gt: # false positive
                    stats[grammeme]['fp'] += 1
                elif in_gt and not in_pred: # false negative
                    stats[grammeme]['fn'] += 1

        self.fscores = {}
        numerator, denominator = 0, 0 # for micro-averaged
        for string, counts in stats.items():
            tp = counts['tp']
            fp = counts['fp']
            fn = counts['fn']

            # precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            # F-score itself
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0

            self.fscores[string] = f1
            numerator += 2 * tp
            denominator += (2 * tp + fp + fn)

        self.fscores['micro-averaged'] = numerator / denominator
        print(f'F-score (micro-averaged):\t{self.fscores["micro-averaged"]:.2f}%')


def get_metrics(parameters):
    evaluator = Evaluator(parameters)

    new_data = evaluator.create_parquet()

    evaluator.calculate_accuracy()
    evaluator.calculate_fscore()

    return evaluator.accuracy, evaluator.fscores


def main():
    parameters = parse_arguments()

    if parameters['seed'] is not None:
        get_metrics(parameters)
        return

    per_run_accuracy = 0.0
    per_run_fscores = defaultdict(int)

    for seed in range(NUMBER_OF_RUNS):
        parameters['seed'] = seed
        accuracy, fscores = get_metrics(parameters)

        per_run_accuracy.append(accuracy)
        for k, v in fscores.items():
            per_run_fscores[k].append(v)

    per_run_accuracy = np.array(per_run_accuracy, dtype=float)
    mean_accuracy = per_run_accuracy.mean()
    std_accuracy = per_run_accuracy.std()
    print(f'Accuracy:\nmean\tstd\n{mean_accuracy}\t{std_accuracy}')

    mean_fscores = dict()
    std_fscores = dict()
    for k, v in per_run_fscores.items():
        per_run_fscores[k] = np.array(v, dtype=float)
        mean_fscores[k] = per_run_fscores[k].mean()
        std_fscores[k] = per_run_fscores[k].std()
        if k == 'micro-averaged':
            print(f'F-score (micro-averaged):\nmean\tstd\n{mean_fscores[k]}\t{std_fscores[k]}')


if __name__ == "__main__":
    main()
