# -*- coding: utf-8 -*-


import gensim


class Word2VecSynonym:
    def __init__(self, word_embedding_path, topn=10):
        print("loading word embedding ......")
        self.word_embedding = gensim.models.KeyedVectors.load_word2vec_format(word_embedding_path, binary=False)
        self.topn = topn
        print("DONE !!!")

    def synonym(self, words):
        try:
            ret = self.word_embedding.most_similar(words, topn=self.topn)
        except:
            ret = []
        return ret


