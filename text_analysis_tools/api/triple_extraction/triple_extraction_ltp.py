# -*- coding: utf-8 -*-

"""
采用ltp进行三元组抽取
根据https://github.com/liuhuanyong/EventTriplesExtraction改编，替换pyltp为ltp;
ltp: https://ltp.readthedocs.io/zh_CN/latest/appendix.html
"""

import os
import pprint
import re
from ltp import LTP



class LtpParser:
    def __init__(self):
        self.ltp = LTP()

    '''语义角色标注'''
    def format_labelrole(self, hidden):
        roles = self.ltp.srl(hidden, keep_empty=False)
        roles_dict = {}
        for role in roles[0]:
            roles_dict[role[0]] = {arg[0]: [arg[0], arg[1], arg[2]] for arg in role[1]}
        return roles_dict

    '''句法分析---为句子中的每个词语维护一个保存句法依存儿子节点的字典'''
    def build_parse_child_dict(self, words, postags, arcs):
        child_dict_list = []
        format_parse_list = []
        for index in range(len(words)):
            child_dict = dict()
            for arc_index in range(len(arcs)):
                if arcs[arc_index][1] == index+1:   # arcs的索引从1开始
                    if arcs[arc_index][2] in child_dict:
                        child_dict[arcs[arc_index][2]].append(arc_index)
                    else:
                        child_dict[arcs[arc_index][2]] = []
                        child_dict[arcs[arc_index][2]].append(arc_index)
            child_dict_list.append(child_dict)
        rely_id = [arc[1] for arc in arcs]  # 提取依存父节点id
        relation = [arc[2] for arc in arcs]  # 提取依存关系
        heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
        for i in range(len(words)):
            # ['ATT', '李克强', 0, 'nh', '总理', 1, 'n']
            a = [relation[i], words[i], i, postags[i], heads[i], rely_id[i]-1, postags[rely_id[i]-1]]
            format_parse_list.append(a)
        return child_dict_list, format_parse_list

    '''parser主函数'''
    def parser_main(self, sentence):
        words, hidden = self.ltp.seg([sentence])
        postags = self.ltp.pos(hidden)
        arcs = self.ltp.dep(hidden)
        words, postags, arcs = words[0], postags[0], arcs[0]
        # print(words, '\n', postags, '\n', arcs)
        child_dict_list, format_parse_list = self.build_parse_child_dict(words, postags, arcs)
        roles_dict = self.format_labelrole(hidden)
        return words, postags, child_dict_list, roles_dict, format_parse_list


class TripleExtraction():
    def __init__(self):
        self.parser = LtpParser()

    '''文章分句处理, 切分长句，冒号，分号，感叹号等做切分标识'''
    def split_sents(self, content):
        return self.parser.ltp.sent_split([content])

    '''利用语义角色标注,直接获取主谓宾三元组,基于A0,A1,A2'''
    def ruler1(self, words, postags, roles_dict, role_index):
        v = words[role_index]
        role_info = roles_dict[role_index]
        if 'A0' in role_info.keys() and 'A1' in role_info.keys():
            s = ''.join([words[word_index] for word_index in range(role_info['A0'][1], role_info['A0'][2] + 1) if
                         postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
            o = ''.join([words[word_index] for word_index in range(role_info['A1'][1], role_info['A1'][2] + 1) if
                         postags[word_index][0] not in ['w', 'u', 'x'] and words[word_index]])
            if s and o:
                return '1', [s, v, o]
        return '4', []

    '''三元组抽取主函数
    关系类型	    Tag	Description	Example
    主谓关系	    SBV	subject-verb	我送她一束花 (我 <– 送)
    动宾关系	    VOB	直接宾语，verb-object	我送她一束花 (送 –> 花)
    间宾关系	    IOB	间接宾语，indirect-object	我送她一束花 (送 –> 她)
    前置宾语	    FOB	前置宾语，fronting-object	他什么书都读 (书 <– 读)
    兼语  	    DBL	double	他请我吃饭 (请 –> 我)
    定中关系	    ATT	attribute	红苹果 (红 <– 苹果)
    状中结构	    ADV	adverbial	非常美丽 (非常 <– 美丽)
    动补结构	    CMP	complement	做完了作业 (做 –> 完)
    并列关系	    COO	coordinate	大山和大海 (大山 –> 大海)
    介宾关系	    POB	preposition-object	在贸易区内 (在 –> 内)
    左附加关系	LAD	left adjunct	大山和大海 (和 <– 大海)
    右附加关系	RAD	right adjunct	孩子们 (孩子 –> 们)
    独立结构	    IS	independent structure	两个单句在结构上彼此独立
    核心关系	    HED	head	指整个句子的核心
    '''
    def ruler2(self, words, postags, child_dict_list, arcs, roles_dict):
        svos = []
        for index in range(len(postags)):
            # print(index)
            tmp = 1
            # 先借助语义角色标注的结果，进行三元组抽取
            if index in roles_dict:
                flag, triple = self.ruler1(words, postags, roles_dict, index)
                if flag == '1':
                    svos.append(triple)
                    tmp = 0
            if tmp == 1:
                # 如果语义角色标记为空，则使用依存句法进行抽取
                if postags[index]:
                    # 抽取以谓词为中心的事实三元组
                    child_dict = child_dict_list[index]
                    # 主谓宾
                    if 'SBV' in child_dict and 'VOB' in child_dict:
                        r = words[index]
                        e1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                        e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                        svos.append([e1, r, e2])

                    # 定语后置，动宾关系
                    relation = arcs[index][0]
                    head = arcs[index][2]
                    if relation == 'ATT':
                        if 'VOB' in child_dict:
                            e1 = self.complete_e(words, postags, child_dict_list, head - 1)
                            r = words[index]
                            e2 = self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                            temp_string = r + e2
                            if temp_string == e1[:len(temp_string)]:
                                e1 = e1[len(temp_string):]
                            if temp_string not in e1:
                                svos.append([e1, r, e2])
                    # 含有介宾关系的主谓动补关系
                    if 'SBV' in child_dict and 'CMP' in child_dict:
                        e1 = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                        cmp_index = child_dict['CMP'][0]
                        r = words[index] + words[cmp_index]
                        if 'POB' in child_dict_list[cmp_index]:
                            e2 = self.complete_e(words, postags, child_dict_list, child_dict_list[cmp_index]['POB'][0])
                            svos.append([e1, r, e2])
        return svos

    '''对找出的主语或者宾语进行扩展'''
    def complete_e(self, words, postags, child_dict_list, word_index, deep=0):
        deep += 1
        if deep == 3:
            return ""
        child_dict = child_dict_list[word_index]
        prefix = ''
        if 'ATT' in child_dict:
            for i in range(len(child_dict['ATT'])):
                prefix += self.complete_e(words, postags, child_dict_list, child_dict['ATT'][i], deep)
        postfix = ''
        if postags[word_index] == 'v':
            if 'VOB' in child_dict:
                postfix += self.complete_e(words, postags, child_dict_list, child_dict['VOB'][0], deep)
            if 'SBV' in child_dict:
                prefix = self.complete_e(words, postags, child_dict_list, child_dict['SBV'][0], deep) + prefix
        return prefix + words[word_index] + postfix

    '''程序主控函数'''

    def triples_main(self, content):
        sentences = self.split_sents(content)
        sentences = [sent for sent in sentences if len(sent.strip()) > 5]
        svos = []
        for sentence in sentences:
            words, postags, child_dict_list, roles_dict, arcs = self.parser.parser_main(sentence)

            # print(words)
            # print(postags)
            # pprint.pprint(child_dict_list)
            # pprint.pprint(roles_dict)
            # pprint.pprint(arcs)

            svo = self.ruler2(words, postags, child_dict_list, arcs, roles_dict)
            svos += svo
        return svos


if __name__ == '__main__':
    text = "履行反洗钱义务的机构及其工作人员依法提交大额交易和可疑交易报告，受法律保护。"
    extractor = TripleExtraction()
    res = extractor.triples_main(text)
    print(text)
    pprint.pprint(res)
    print("---------"*5)


