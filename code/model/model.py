"""Contains class Model that has encoder and decoder and produces predictions and probabilities of grammemes."""

import torch
import torch.nn as nn

from model.layers.encoder import LSTMEncoder, RoBERTaEncoder, RuBERTEncoder
from model.layers.decoder import Decoder


class Model(nn.Module):
    """Contains encoder and decoder.

    Args:
        conf (dict): Dictionary with configuration parameters.
        vocab (Vocab): Instance of class containing vocab.

    Attributes:
        encoder: Encoder class from encoder.py
        decoder: Decoder class from decoder.py
    """

    def __init__(self, conf, vocab, encoder_type, **kwargs):
        super().__init__()
        self.conf = conf
        self.vocab = vocab
        self.encoder_type = encoder_type.lower()

        if self.encoder_type == 'lstm':
            self.encoder = LSTMEncoder(self.conf, self.vocab)
        elif self.encoder_type == 'roberta':
            self.encoder = RoBERTaEncoder(self.conf, self.vocab)
        elif self.encoder_type == 'rubert':
            self.encoder = RuBERTEncoder(self.conf, self.vocab)
        else:
            raise ValueError(f'Unknown encoder {self.encoder_type}')

        self.decoder = Decoder(self.conf, self.vocab)


    def forward(self, words_batch, chars_batch, labels_batch=None):
        """Uses Encoder and Decoder to perform one pass on a sinle batch.

        Args:
            words_batch (torch.Tensor): Tensor of words indices for every word in a batch.
                Size (max_sentence_length, batch_size).
            labels_batch (torch.Tensor, default None): Tensor of labels indices for every word in a batch.
                Size (max_label_length, max_sentence_length * batch_size).
                If None, decoder will use generated grammemes for the next prediction (inference mode).
                Otherwise, decoder will use labels_batch (training mode).

        Returns:
            tuple: Tuple consists of predicted grammemes and their probabilities.
        """

        # shape (max_sentence_length * batch_size, grammeme_LSTM_hidden)
        # encoder_output is a single tensor
        encoder_output = self.encoder(words_batch=words_batch, chars_batch=chars_batch)
        decoder_hidden, decoder_cell = encoder_output

        predictions, probabilities = self.decoder(decoder_hidden, decoder_cell, labels_batch)
        return predictions, probabilities
