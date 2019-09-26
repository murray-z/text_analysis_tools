# -*- coding: utf-8 -*-

import os
import re
import jieba
import json
from xpinyin import Pinyin
from collections import Counter
from text_analysis_tools.api.spell_correct import STOPWORDS


stopwords_path = STOPWORDS


class SpellCorrect:
    def __init__(self, corpus_path, train=True, ret_num=10, model_dir="spell_correct_model"):
        """
        :param corpus_path: 文本路径
        :param train: 是否根据文本，生成纠错词库，默认文本为《水浒传》
        :param ret_num: 返回可能的纠错结果数目
        """
        self.p = Pinyin()
        self.corpus_path = corpus_path
        self.ret_num = ret_num
        self.model_dir = model_dir

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        self.pinyin_word_path = os.path.join(self.model_dir, 'pinyin_word.json')
        self.word_count_path = os.path.join(self.model_dir, 'word_count.json')

        if train:
            self.split_text()

        self.pinyin_word = self.load_json(self.pinyin_word_path)
        self.word_count = self.load_json(self.word_count_path)
        self.WORDS = self.pinyin_word.keys()

    def load_stopwords(self):
        with open(stopwords_path, encoding='utf-8') as f:
            return [line.strip() for line in f]

    def split_text(self):
        pinyin_word = {}  # 存放拼音对应词
        stopwords = self.load_stopwords()
        with open(self.corpus_path, encoding='utf-8') as f:
            words = jieba.lcut(re.sub("[\sa-z0-9\r\n]+", "", f.read()))
            word_count = dict(Counter(words))  # 词对应词频
            for key in word_count:
                if key not in stopwords:
                    pinyin = self.p.get_pinyin(key, splitter='')
                    if pinyin not in pinyin_word:
                        pinyin_word[pinyin] = [key]
                    else:
                        pinyin_word[pinyin].append(key)
        self.dump_json(word_count, self.word_count_path)
        self.dump_json(pinyin_word, self.pinyin_word_path)

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def load_json(self, json_file_path):
        with open(json_file_path, encoding='utf-8') as f:
            return json.loads(f.read(), encoding='utf-8')

    def dump_json(self, content, json_file_path):
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=4)

    def correct(self, word):
        word_pinyin = self.p.get_pinyin(word, splitter='')
        candidate_pinyin = self.candidates(word_pinyin)
        ret_dic = {}
        words = []
        for pinyin in candidate_pinyin:
            words.extend(self.pinyin_word[pinyin])
        for word in words:
            ret_dic[word] = self.word_count.get(word, 0)
        sort_word = sorted(ret_dic.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sort_word[:self.ret_num]]


if __name__ == '__main__':
    spell_correct = SpellCorrect()
    print(spell_correct.correct('宋江'))
    print(spell_correct.correct('松江'))
    print(spell_correct.correct('李奎'))
    print(spell_correct.correct('吴宋'))
    print(spell_correct.correct('送三连'))



