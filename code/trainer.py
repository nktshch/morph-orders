"""Docstring for trainer.py"""

from data_preparation.sampler import BucketSampler

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from scipy.optimize import linear_sum_assignment as lsa
from tqdm import tqdm


class Trainer(nn.Module):
    """Class performs training. More info will be added later.

    Args:
        conf (dict): Dictionary with configuration parameters.
        model (Model): Instance of class containing model parameters.
        train_data (dataset): Dataset for training.
        valid_data (dataset): Dataset for validation.
        run_number: Run / random seed number.
        subset_size (float or int): The part of dataset from model.data that will be used.
            If int, treated as the number of samples from model.data with 0 treated as a whole dataset.
            If float, should be between 0 and 1, treated as the proportion of the dataset used during training.
        save_model (bool): If True, best model is saved in folder. If False, only logs are saved.
            Use False for hyperparameters selection.
    """

    def __init__(self, conf, model, train_data, valid_data, run_number=0, subset_size=0, save_model=True):
        super().__init__()
        self.conf = conf
        self.model = model
        self.vocab = model.vocab
        self.run_number = run_number
        self.save_model = save_model

        if subset_size == 0:
            train_subset = train_data
            valid_subset = valid_data
        elif isinstance(subset_size, int) and subset_size > 0:
            train_subset = subset_from_dataset(train_data, subset_size)
            valid_subset = subset_from_dataset(valid_data, subset_size)
        elif isinstance(subset_size, float) and 0.0 < subset_size < 1.0:
            train_subset = subset_from_dataset(train_data, int(subset_size * len(train_data)))
            valid_subset = subset_from_dataset(valid_data, int(subset_size * len(valid_data)))
        else:
            raise TypeError(f'subset_size should be a non-negative int or a float between 0 and 1, not {subset_size}')

        if self.conf['bucket_train_data']:
            sampler = BucketSampler(train_subset, self.conf['sentence_train_batch_size'])
        else:
            sampler = None

        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.conf['sentence_train_batch_size'],
            collate_fn=lambda item: collate_fn(item, self.vocab.vocab['grammeme-index'], self.vocab.sorting_order),
            sampler=sampler)
        self.valid_loader = DataLoader(
            valid_subset,
            batch_size=self.conf['sentence_eval_batch_size'],
            collate_fn=lambda item: collate_fn(item, self.vocab.vocab['grammeme-index'], self.vocab.sorting_order)) if valid_subset else []

        self.xe_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        self.nll_loss = nn.NLLLoss(reduction='mean', ignore_index=0)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.conf['learning_rate'])

        self.writer = SummaryWriter(log_dir=f'{self.conf["model_folder"]}/logs/seed_{self.run_number}')
        self.current_epoch = 0
        self.best_accuracy = 0
        self.best_loss = np.inf
        self.no_improv = 0

        # print(f'{len(self.train_loader)} batches in train')
        # print(f'{len(self.valid_loader)} batches in valid')


    def train(self, **kwargs):
        """Main method that performs training."""

        for epoch in range(self.conf['max_epochs']):
            self.model.train()
            self.train_epoch()

            if len(self.valid_loader) > 0:
                self.model.eval()
                with torch.no_grad():
                    valid_accuracy, valid_loss = self.valid_epoch()
                if valid_loss >= self.best_loss:
                    self.no_improv += 1
                    if self.no_improv >= self.conf['no_improv']:
                        print(f'No improvement in accuracy for {self.conf["no_improv"]} epochs, stopping early')
                        break
                else:
                    self.no_improv = 0
                    self.best_accuracy = valid_accuracy
                    self.best_loss = valid_loss
                    if self.save_model is True:
                        torch.save([self.conf, self.model.state_dict()],
                                   f'{self.conf["model_folder"]}/model_{self.run_number}.pt')
                self.current_epoch += 1


    def train_epoch(self):
        """One train epoch on full dataset.

        1. train_loader gives a batch
        2. it is fed into the model that gives probabilities for each grammeme
        3. the loss is then computed and weights are updated
        4. this loop repeats until whole train dataset is done
        """
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), colour='#bbbb88')
        # current_epoch_tags = [] # stores tags for this epoch

        for iteration, (words_batch, labels_batch) in progress_bar:
            # words_batch = words_batch.to(self.conf['device'])
            # labels_batch = labels_batch.to(self.conf['device'])
            labels_batch = torch.tensor(labels_batch, dtype=torch.long).to(self.conf['device'])
            labels_batch = labels_batch.view(-1, labels_batch.shape[2]).permute(1, 0)

            self.optimizer.zero_grad()
            _, probabilities = self.model(words_batch, labels_batch)
            targets = labels_batch[1:]  # slice is taken to ignore SOS token
            probabilities = probabilities[:len(targets)]
            # probabilities has shape (max_label_length, max_sentence_length * batch_size, grammemes_in_vocab)
            # targets has shape (max_label_length, max_sentence_length * batch_size)

            targets = targets.permute(1, 0)
            probabilities = probabilities.permute(1, 2, 0)

            if self.conf['loss'] == 'xe':
                loss = self.xe_loss(probabilities, targets)
            elif self.conf['loss'] == 'oaxe':
                loss, _ = self.oaxe_loss(probabilities, targets)
            else:
                raise ValueError(f'Unknown loss: {self.conf["loss"]}')

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf['clip'])
            self.optimizer.step()

            if iteration % self.conf['logger_frequency'] == 0:
                self.writer.add_scalar('training loss',
                                       loss.item(),
                                       self.current_epoch * len(self.train_loader) + iteration)


    def valid_epoch(self):
        """The same as train epoch except weights are not updated."""

        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), colour='#bbbbff')
        running_error = 0.0
        correct, total = 0, 0

        for iteration, (words_batch, labels_batch) in progress_bar:
            # words_batch = words_batch.to(self.conf['device'])
            # labels_batch = labels_batch.to(self.conf['device'])
            labels_batch = torch.tensor(labels_batch, dtype=torch.long).to(self.conf['device'])
            labels_batch = labels_batch.view(-1, labels_batch.shape[2]).permute(1, 0)


            predictions, probabilities = self.model(words_batch, None)
            targets = labels_batch[1:]  # slice is taken to ignore SOS token
            probabilities = probabilities[:len(targets)]
            # probabilities has shape (max_label_length, max_sentence_length * batch_size, grammemes_in_vocab)
            # targets has shape (max_label_length, max_sentence_length * batch_size)

            targets = targets.permute(1, 0)
            predictions = predictions.permute(1, 0)
            probabilities = probabilities.permute(1, 2, 0)

            if self.conf['loss'] == 'xe':
                error = self.xe_loss(probabilities, targets)
            elif self.conf['loss'] == 'oaxe':
                error, best_permutations = self.oaxe_loss(probabilities, targets)
                targets = torch.gather(targets, 1, best_permutations)
            else:
                raise ValueError(f'Unknown loss: {self.conf["loss"]}')

            correct_batch, total_batch = calculate_accuracy(self.vocab, self.conf, predictions, targets)
            correct += correct_batch
            total += total_batch
            running_error += error.item() * total_batch

        valid_accuracy = correct / total
        valid_loss = running_error / total
        self.writer.add_scalar('valid accuracy',
                               valid_accuracy,
                               (self.current_epoch + 1) * len(self.valid_loader))
        self.writer.add_scalar('valid loss (average)',
                               valid_loss,
                               (self.current_epoch + 1) * len(self.valid_loader))
        print(f'[EPOCH {self.current_epoch}] acc: {valid_accuracy:.5}, loss: {valid_loss:.5}')
        return valid_accuracy, valid_loss


    def oaxe_loss(self, probabilities_, targets):
        """
        Calculates Order-Agnostic Cross Entropy Loss
        """
        # probabilities has shape (max_sentence_length * batch_size, grammemes_in_vocab, max_label_length)
        # targets has shape (max_sentence_length * batch_size, max_label_length)
        probabilities = self.logsoftmax(probabilities_)

        best_permutations_ = torch.tensor(range(targets.shape[1])).to(targets).repeat((targets.shape[0], 1))
        for i in range(targets.shape[0]):  # for 1 sequence in a batch
            target_row = targets[i]
            n_nonpad = target_row.ne(0).sum()  # to exclude padding (0 is pad_id)
            if n_nonpad == 0:  # skip if row is padding itself
                continue
            if self.conf['same_length'] is False:
                n_nonpad -= 1 # in this case we should exclude eos token

            target_row = target_row[:n_nonpad] # remove padding
            probabilities_matrix = probabilities[i]  # probabilities for this sequence (grammemes_in_vocab, max_label_length)
            probabilities_matrix = probabilities_matrix[:, :n_nonpad] # remove probabilities on padded labels

            cost_matrix = -probabilities_matrix[target_row]  # get cost matrix for lsa
            cost_matrix_numpy = cost_matrix.detach().cpu().numpy()
            # permute because of how lsa works, we need rows indices, not column

            row_indices, col_indices = lsa(cost_matrix_numpy)
            best_permutations_[i, :n_nonpad] = torch.tensor(np.argsort(col_indices)).to(targets)

        # best_permutations = best_permutations_[:, None, :]
        # best_permutations = best_permutations.expand_as(probabilities)

        new_targets = torch.gather(targets, 1, best_permutations_)

        smooth_loss = self.nll_loss(probabilities, new_targets)  # can be computed with regular negative log likelihood
        return smooth_loss, best_permutations_


def calculate_accuracy(vocabulary, conf, predictions, targets):
    """Metrics is a ratio of correctly predicted tags to total number of tags.

    All grammemes in a tag must be predicted correctly for it to count as correct.
    """

    n_total, n_correct = 0, 0
    masked_predictions = masked_select(predictions, vocabulary.vocab['grammeme-index'][conf['EOS']])

    iteration_mask = targets[:, 0].to(torch.bool)
    for tag, target in zip(masked_predictions[iteration_mask], targets[iteration_mask]):
        n_total += 1
        tag_nonzero = tag[tag.nonzero()]
        target_nonzero = target[target.nonzero()]
        equal = torch.equal(tag_nonzero, target_nonzero)
        n_correct += int(equal)

    return n_correct, n_total


def collate_fn(batch, grammemes_vocab, sorting_order, pad='$PAD$', sos='$SOS$', eos='$EOS$', unk='$UNK$'):
    """Collate method for BERT-like encoders.

    Args:
        batch (dict): Batch returned by dataset
        grammemes_vocab (dict): 'grammeme-index' dict from vocab
        sorting_order (dict): 'sorting_order' dict from vocab. Used to sort grammemes according to selected order
        pad (str): padding grammeme
        sos (str): start of sequence grammeme
        eos (str): end of sequence grammeme
        unk (str): unknown grammeme
    """

    # dataset returns dict with keys 'id', 'tokens', 'tags'
    # batch is dict that stores a list for each key
    tokens = [item['tokens'] for item in batch]
    tags = [[tag.split('|') for tag in item['tags']] for item in batch]
    max_sentence_length = max(map(len, tokens))
    max_label_length = 2 + max([max(map(len, item)) for item in tags])

    new_tags = []
    for item in tags:
        item_indices = []
        for tag in item:
            ordered_tag = sorted(tag, key=lambda g: sorting_order.get(g, len(sorting_order)))
            indices = [grammemes_vocab.get(g, grammemes_vocab[unk]) for g in ordered_tag]
            indices.insert(0, grammemes_vocab[sos])
            indices.append(grammemes_vocab[eos])
            indices.extend([grammemes_vocab[pad]] * (max_label_length - len(indices)))
            item_indices += [indices]
        item_indices.extend([[grammemes_vocab[pad]] * max_label_length] * (max_sentence_length - len(item)))
        new_tags.append(item_indices)

    # use code below to check

    # for token, item in zip(tokens, new_tags):
    #     print(token)
    #     for tag in item:
    #         print(tag)

    return tokens, new_tags


def subset_from_dataset(data, n):
    """Outputs first n entries from data as another dataset."""
    subset = Subset(data, range(n)) if data else None
    return subset


def masked_select(a, value):
    """Zero all elements that come after a given value in a row. Used for zeroing elements after EOS token.

    Args:
        a (torch.Tensor): Input tensor.
        value (a.dtype): Value after which all elements should be equal to zero (EOS token index).

    Returns:
        torch.Tensor: Masked tensor of the same shape as a.

    Examples:
        >>> input_tensor = torch.Tensor([[1., 2., 3., 4., 99., 5., 2., 1.],
                              [1., 99., 99., 4., 3., 5., 99., 3.],
                              [1., 3., 3., 4., 1., 5., 2., 1.]])
        >>> print(masked_select(input_tensor, 99))
        tensor([[ 1.,  2.,  3.,  4., 99.,  0.,  0.,  0.],
                [ 1., 99.,  0.,  0.,  0.,  0.,  0.,  0.],
                [ 1.,  3.,  3.,  4.,  1.,  5.,  2.,  1.]])
    """

    first_occurrence = (a == value).cumsum(dim=1)
    first_occurrence = first_occurrence.to(torch.bool)
    padding = torch.zeros((a.shape[0], 1), dtype=torch.bool).to(a.device)
    narrow = torch.narrow(first_occurrence, 1, 0, a.shape[1] - 1)
    mask = torch.hstack((padding, narrow))
    return a * (~mask)
