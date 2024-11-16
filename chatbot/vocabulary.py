import os
import json
import codecs
import csv
import itertools
import re
from utils import normalizeString
from config import PAD_token, SOS_token, EOS_token, MAX_LENGTH

class Voc:
    """ Vocabulary class for mapping words to indexes """
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)

    def indexesFromSentence(self, sentence):
        return [self.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def load_json_conversations(corpus_path):
    utterances = []
    conversations = {}
    
    with open(os.path.join(corpus_path, 'utterances.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            utterance = json.loads(line)
            conv_id = utterance['conversation_id']
            
            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append(utterance)
    
    return conversations

def extract_pairs(conversations):
    pairs = []
    for conv_id, utterances in conversations.items():
        utterances.sort(key=lambda x: x['id'])
        for i in range(len(utterances) - 1):
            input_text = normalizeString(utterances[i]['text'])
            target_text = normalizeString(utterances[i + 1]['text'])
            if input_text and target_text:
                pairs.append([input_text, target_text])
    return pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def loadPrepareData(corpus_path):
    print("Loading conversations...")
    conversations = load_json_conversations(corpus_path)
    print("Extracting pairs...")
    pairs = extract_pairs(conversations)
    print("Read {!s} sentence pairs".format(len(pairs)))
    
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    
    voc = Voc("movie-corpus")
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    
    return voc, pairs