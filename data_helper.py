# -*- coding: utf-8 -*-

import jieba
import json


def change_cls_data_to_fasttext(raw_data_path, fasttext_path):
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        with open(fasttext_path, 'w', encoding='utf-8') as f_w:
            for line in f:
                lis = line.strip().split('\t')
                f_w.write("__label__{} {}\n".format(lis[0], " ".join(jieba.lcut(lis[1]))))

def make_synonym_dict(text_file_path, text_file_path2, json_file_path):
    synonym_dict = {}

    with open(text_file_path, encoding='utf-8') as f_r:
        for line in f_r:
            lis = line.strip().split('        ')
            if len(lis) > 1:
                synonym_dict[lis[0]] = lis[1].split('、')

    with open(text_file_path2, encoding='utf-8') as f_r:
        for line in f_r:
            lis = line.strip().split()
            if lis[0] in synonym_dict.keys():
                synonym_dict[lis[0]].append(lis[1])
            else:
                synonym_dict[lis[0]] = [lis[1]]

    ret = {}
    for key, val in synonym_dict.items():
        ret[key] = list(set(val))
    with open(json_file_path, 'w', encoding='utf-8') as f_w:
        json.dump(ret, f_w, indent=4, ensure_ascii=False)

    print(ret)



if __name__ == "__main__":
    # change_cls_data_to_fasttext("./test_data/test_classification.txt", "./test_data/test_fasttext_cls.txt")
    make_synonym_dict('./test_data/synonym.txt', './test_data/merge_syno.txt', './text_analysis_tools/api/data/synonym.json')

