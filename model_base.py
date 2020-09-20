#######
# Copyright 2020 Jian Zhang, All rights reserved
##
import logging
from abc import ABC, abstractmethod

class ModelBase(ABC):
    @abstractmethod
    def __init__(self, load_trained=False):
        '''
        :param load_trained
            If load_trained is True, load a trained model from a file.
            Should include code to download the file from Google drive if necessary.
            else, construct the model
        '''
        if load_trained:
            logging.info('load model from file ...')

    @abstractmethod
    def train(self, x):
        '''
        Train the model on one batch of data
        :param x: train data. For composer training, a single torch tensor will be given
        and for critic training, x will be a tuple of two tensors (data, label)
        :return: (mean) loss of the model on the batch
        '''
        pass

class ComposerBase(ModelBase):
    '''
    Class wrapper for a model that can be trained to generate music sequences.
    '''
    @abstractmethod
    def compose(self, n):
        '''
        Generate a music sequence
        :param n: length of the sequence to be generated
        :return: the generated sequence
        '''
        pass

class CriticBase(ModelBase):
    '''
    Class wrapper for a model that can be trained to criticize music sequences.
    '''
    @abstractmethod
    def score(self, x):
        '''
        Compute the score of a music sequence
        :param x: a music sequence
        :return: the score between 0 and 1 that reflects the quality of the music: the closer to 1, the better
        '''
        pass
