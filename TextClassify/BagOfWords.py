# -*- coding: utf-8 -*-

import os
import re
import jieba
import numpy
from scipy import sparse

class BagOfWords:
    def __init__(self, dir):
        self.dir = dir
        jieba.set_dictionary('../dictionary/dict.txt.big')

        #load sentiment dictionary
        jieba.load_userdict('../dictionary/SD/HOWNET_Evaluation_Negative.txt')
        jieba.load_userdict('../dictionary/SD/HOWNET_Evaluation_Positive.txt')
        jieba.load_userdict('../dictionary/SD/HOWNET_Perception.txt')
        jieba.load_userdict('../dictionary/SD/HOWNET_Sentiment_Negative.txt')
        jieba.load_userdict('../dictionary/SD/HOWNET_Sentiment_Positive.txt')
        jieba.load_userdict('../dictionary/SD/NTUSD_Negative.txt')
        jieba.load_userdict('../dictionary/SD/NTUSD_Positive.txt')
        jieba.load_userdict('../dictionary/SD/iMFinanceSD_negative.txt')
        jieba.load_userdict('../dictionary/SD/iMFinanceSD_positive.txt')

        #load MDJ dictionary
        jieba.load_userdict('../dictionary/MDJ/MDJ_Common.v02.txt')
        jieba.load_userdict('../dictionary/MDJ/MDJFD.v02.txt')
        jieba.load_userdict('../dictionary/MDJ/MDJFD.eng.v02.txt')

        #load extend dictionary
        jieba.load_userdict('../dictionary/THUOCL/THUOCL_caijing_cht.txt')

        #load name entity
        jieba.load_userdict('../dictionary/NE/Countries.txt')
        jieba.load_userdict('../dictionary/NE/Organization.txt')
        jieba.load_userdict('../dictionary/NE/President.txt')

        #load wordnet dictionary
        jieba.load_userdict('../dictionary/WN/wordnet.txt')
        print('User Dict loaded.')

    def build_dictionary(self):
        dict_set = set()
        count = 0
        for (dirname, dirs, files) in os.walk(self.dir):
            for file in files:
                if file.endswith('.txt'):
                    filename = os.path.join(dirname, file)
                    with open(filename, 'rb') as f:
                        count += 1
                        for line in f:
                            line = self.process_line(line)
                            words = jieba.cut(line.strip(), cut_all=False)
                            dict_set |= set(words)
        self.num_samples = count
        self.dict = self.reduce_dict(dict_set)

    def load_dictionary(self, dir):
        #import cPickle as Pickle
        import _pickle as Pickle
        try:
            print("loaded dictionary from %s" % dir)
            self.dict = Pickle.load(open(dir, 'rb'))
            print("done")
        except IOError:
            print("error while loading from %s" % dir)

    def save_dictionary(self, dir):
        import _pickle as Pickle
        Pickle.dump(self.dict, open(dir, 'wb'))
        print("saved dictionary to %s" % dir)

    def save_dictionary2Json(self, dir):
        import json
        json.dump(self.dict, open(dir, 'w'), ensure_ascii=False)
        print("saved dictionary to %s" % dir)

    def reduce_dict(self, dict_set):
        dict_copy = dict_set.copy()
        for word in dict_set:
            if len(word) < 2:
                dict_copy.remove(word)
            else:
                try:
                    float(word)
                    dict_copy.remove(word)
                except ValueError:
                    continue
        dictionary = {}
        for idx, word in enumerate(dict_copy):
            dictionary[word] = idx
        return dictionary

    def process_line(self, line):
        line = line.decode("utf8")
        return re.sub("]-·[\s+\.\!\/_,$%^*(+\"\':]+|[+——！，。？、~@#￥%……&*（）():\"=《]+",
                                           " ", line)

    def transform_data(self, dir, partial=1, balance=False):
        import random
        from scipy import sparse
        print("transforming data in to bag of words vector")
        data = []
        target = []
        count = 0
        n_balance = 0
        if partial > 1 or partial <=0:
            partial = 1
        partial = int(partial * 10) / 10

        ## Find least Sample
        if balance:
            for (dirname, dirs, files) in os.walk(dir):
                n_files = len(files)
                if n_files > 0:
                    if n_balance == 0:
                        n_balance = n_files
                        continue
                    if n_files < n_balance:
                        n_balance = n_files
                else:
                    continue

        for (dirname, dirs, files) in os.walk(dir):
            if balance and len(files) >= n_balance:
                the_files = random.sample(files, int(n_balance * partial))
            else:
                the_files = random.sample(files, int(len(files) * partial))
            #for file in files:
            for file in the_files:
                if file.endswith('.txt'):
                    count += 1
                    filename = os.path.join(dirname, file)
                    tags = re.split('[/\\\\]', dirname)
                    tag = tags[-1]
                    word_vector = numpy.zeros(len(self.dict))
                    with open(filename, 'rb') as f:
                        for line in f:
                            line = self.process_line(line)
                            words = jieba.cut(line.strip(), cut_all=False)
                            for word in words:
                                try:
                                    word_vector[self.dict[word]] += 1
                                except KeyError:
                                    pass
                    #data.append(sparse.csr_matrix(word_vector))
                    data.append(word_vector)
                    target.append(tag)
        self.num_samples = count
        print("done")
        return sparse.csr_matrix(numpy.asarray(data)),numpy.asarray(target)

    def trainsorm_single_file(self, file):
        word_vector = numpy.zeros(len(self.dict))
        with open(file, 'rb') as f:
            for line in f:
                line = self.process_line(line)
                words = jieba.cut(line.strip(), cut_all=False)
                for word in words:
                    try:
                        word_vector[self.dict[word]] += 1
                    except KeyError:
                        pass
        return word_vector

    def saveModel(self, model, dir):
        import _pickle as Pickle
        Pickle.dump(model, open(dir, 'wb'))
        print("Saved model to %s" % dir)

    def loadModel(self, dir):
        import _pickle as Pickle
        print("Load model from %s" % dir)
        return Pickle.load(open(dir, 'rb'))