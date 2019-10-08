# -*- coding: utf-8 -*-

import jieba
import json


def change_cls_data_to_fasttext(raw_data_path, fasttext_path):
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        with open(fasttext_path, 'w', encoding='utf-8') as f_w:
            for line in f:
                lis = line.strip().split('\t')
                f_w.write("__label__{} {}\n".format(lis[0], " ".join(jieba.lcut(lis[1]))))

def make_synonym_dict(text_file_path, json_file_path):
    synonym_dict = {}
    with open(json_file_path, 'w', encoding='utf-8') as f_w:
        with open(text_file_path, encoding='utf-8') as f_r:
            for line in f_r:
                lis = line.strip().split('        ')
                print(lis)
                if len(lis) > 1:
                    synonym_dict[lis[0]] = lis[1].split('、')
        json.dump(synonym_dict, f_w, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    # change_cls_data_to_fasttext("./test_data/test_classification.txt", "./test_data/test_fasttext_cls.txt")
    make_synonym_dict('./test_data/synonym.txt', './text_analysis_tools/api/data/synonym.json')
