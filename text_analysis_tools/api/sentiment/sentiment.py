# -*- coding: utf-8 -*-


import os
import json
import jieba.analyse
import jieba


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sentiment_path = os.path.join(CURRENT_PATH, 'data', 'sentimentDict.json')
stopwords_path = os.path.join(CURRENT_PATH, 'data', 'stopwords.txt.json')
degree_path = os.path.join(CURRENT_PATH, 'data', 'degreeDict.json')
not_path = os.path.join(CURRENT_PATH, 'data', 'notDict.json')
jieba_dic_path = os.path.join(CURRENT_PATH, 'data', 'jieba.dic')

# 加载情感词典
jieba.load_userdict(jieba_dic_path)


class SentimentAnalysis():
    def __init__(self):
        self.sentiment_score_dic = self.load_json(sentiment_path)
        self.degree_score = self.load_json(degree_path)
        self.notwords = self.load_json(not_path)

    def load_json(self, json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.loads(f.read(), encoding='utf-8')

    def analysis(self, sentence):
        words = jieba.lcut(sentence)
        score = self.sentiment_score_dic.get(words[0], 0)
        if len(words) > 1:
            score += self.sentiment_score_dic.get(words[1], 0) * self.notwords.get(words[0], 1) * self.degree_score.get(words[0], 1)
            if len(words) > 2:
                for i in range(2, len(words)):
                    score += self.sentiment_score_dic.get(words[i], 0) * self.notwords.get(words[i-1], 1) * \
                             self.degree_score.get(words[i-1], 1) * self.degree_score.get(words[i-2], 1) * \
                             self.notwords.get(words[i-2], 1)
        if score < 0:
            return {'negative': score}
        if score > 0:
            return {'positive': score}
        return {'middle': score}