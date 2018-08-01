from __future__ import division, absolute_import

import torch
import re, os, sys

from nltk import word_tokenize
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings("ignore")

# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = {
"ain't": "0amnot",
"aren't": "0arenot",
"can't": "0cannot",
"can't've": "0cannothave",
"'cause": "0because",
"could've": "0couldhave",
"couldn't": "0couldnot",
"couldn't've": "0couldnothave",
"didn't": "0didnot",
"doesn't": "0doesnot",
"don't": "0donot",
"hadn't": "0hadnot",
"hadn't've": "0hadnothave",
"hasn't": "0hasnot",
"haven't": "0havenot",
"he'd": "0hewould",
"he'd've": "0hewouldhave",
"he'll": "0hewill",
"he's": "0heis",
"how'd": "0howdid",
"how'll": "0howwill",
"how's": "0howis",
"i'd": "0iwould",
"i'll": "0iwill",
"i'm": "0iam",
"i've": "0ihave",
"isn't": "0isnot",
"it'd": "0itwould",
"it'll": "0itwill",
"it's": "0itis",
"let's": "0letus",
"ma'am": "0madam",
"mayn't": "0maynot",
"might've": "0mighthave",
"mightn't": "0mightnot",
"must've": "0musthave",
"mustn't": "0mustnot",
"needn't": "0neednot",
"oughtn't": "0oughtnot",
"shan't": "0shallnot",
"sha'n't": "0shallnot",
"she'd": "0shewould",
"she'll": "0shewill",
"she's": "0sheis",
"should've": "0shouldhave",
"shouldn't": "0shouldnot",
"that'd": "0thatwould",
"that's": "0thatis",
"there'd": "0therehad",
"there's": "0thereis",
"they'd": "0theywould",
"they'll": "0theywill",
"they're": "0theyare",
"they've": "0theyhave",
"wasn't": "0wasnot",
"we'd": "0wewould",
"we'll": "0wewill",
"we're": "0weare",
"we've": "0wehave",
"weren't": "0werenot",
"what'll": "0whatwill",
"what're": "0whatare",
"what's": "0whatis",
"what've": "0whathave",
"where'd": "0wheredid",
"where's": "0whereis",
"who'll": "0whowill",
"who's": "0whois",
"won't": "0willnot",
"wouldn't": "0wouldnot",
"you'd": "0youwould",
"you'll": "0youwill",
"you're": "0youare"
}

puncts = {
    '.' : '<PERIOD>',
    ',' : '<COMMA>',
    '"' : '<QUOTATION_MARK>',
    ';' : '<SEMICOLON>',
    '!' : '<EXCLAMATION_MARK>',
    '?' : '<QUESTION_MARK>',
    '(' : '<LEFT_PAREN>',
    ')' : '<RIGHT_PAREN>',
    '--': '<HYPHENS>',
    '-' : '<HYPHEN>',
    '\n': '<NEW_LINE>',
    ':' : '<COLON>',
    '—' : '<DASH>'
}


def encode_punc(words):
    new_words = []
    for word in words:
        if not word in puncts:
            new_words.append(word)
        else:
            new_words.append(puncts[word])
    return new_words

def encode_contractions(text):
    text  = BeautifulSoup(text).get_text()
    text = text.lower()

    for word in contractions:
        text = text.replace(word, contractions[word])
    return text.strip()


class Dictionary(object):

    def __init__(self):
        super(Dictionary, self).__init__()
        self.word2idx = {}
        self.idx2word = {}
        self.curr     = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.curr
            self.idx2word[self.curr] = word
            self.curr += 1

    def __len__(self):
        return len(self.word2idx)

class Corpus(object):

    def __init__(self, path='./data'):
        super(Corpus, self).__init__()
        self.dictionary = Dictionary()
        self.train = os.path.join(path,'train.txt')
        self.test = os.path.join(path, 'test.txt')

    def get_data(self, path, batch_size=32):
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words =  encode_punc(word_tokenize(encode_contractions(line))) + ['<EOS>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)


        int_text = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words =  encode_punc(word_tokenize(encode_contractions(line))) + ['<EOS>']
                for word in words:
                    int_text[token] = self.dictionary.word2idx[word]
                    token += 1

        num_batches = int_text.size(0) // batch_size
        int_text = int_text[:num_batches * batch_size]
        return int_text.view(batch_size, -1)

# TODO: Pretrained Embeddings
