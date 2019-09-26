# -*- coding: utf-8 -*-

import jieba


def change_cls_data_to_fasttext(raw_data_path, fasttext_path):
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        with open(fasttext_path, 'w', encoding='utf-8') as f_w:
            for line in f:
                lis = line.strip().split('\t')
                f_w.write("__label__{} {}\n".format(lis[0], " ".join(jieba.lcut(lis[1]))))


if __name__ == "__main__":
    change_cls_data_to_fasttext("./test_data/test_classification.txt", "./test_data/test_fasttext_cls.txt")
