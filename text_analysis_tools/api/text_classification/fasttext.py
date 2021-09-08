# -*- coding: utf-8 -*-

import os
import sys
import fasttext.FastText as ff


class Fasttext:
    def __init__(self, save_model_path=None, train_data_path=None, test_data_path=None, type="train",
                 k=1, threshold=0.0, epoch=5, pretrainedVectors="", label="__label__",
                 lr=0.1, dim=100):
        """
        fasttext
        :param save_model_path: 模型保存路径
        :param train_data_path: 训练样本路径
        :param test_data_path: 测试样本路径
        :param type: 模式：“train/predict”
        :param k: 返回结果个数
        :param threshold: 阈值
        :param epoch: 训练轮数
        :param pretrainedVectors: 预训练词向量路径
        :param label: 标签前缀
        :param lr: 学习率
        :param dim: 词向量维度
        """
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.save_model_path = save_model_path
        self.type = type
        self.k = k
        self.threshold = threshold
        self.epoch = epoch
        self.pretrainedVectors = pretrainedVectors
        self.label = label
        self.lr = lr
        self.dim = dim

        if self.type == "predict":
            if not os.path.exists(self.save_model_path):
                print("MODEL: {} is not EXIST ....")
                sys.exit()
            print("LOAD MODEL FROM: {}".format(self.save_model_path))
            self.classifier = ff.load_model(self.save_model_path)
        else:
            self.classifier = None

    def train(self):
        classifier = ff.train_supervised(input=self.train_data_path, label="__label__", epoch=self.epoch,
                                         pretrainedVectors=self.pretrainedVectors, lr=self.lr, dim=self.dim)
        classifier.save_model(self.save_model_path)
        train_result = classifier.test(self.train_data_path)

        print("### TRAIN RESULT ###")
        print("Train Samples: {}".format(train_result[0]))
        print("Train Precision: {}".format(train_result[1]))
        print("Train Recall: {}\n\n".format(train_result[2]))

        if self.test_data_path:
            test_result = classifier.test(self.test_data_path)

            print("### TEST RESULT ###")
            print("Test Samples: {}".format(test_result[0]))
            print("Test Precision: {}".format(test_result[1]))
            print("Test Recall: {}\n\n".format(test_result[2]))

        print("model save to {}".format(self.save_model_path))

    def predict(self, list_str):
        result = self.classifier.predict(list_str, k=self.k, threshold=self.threshold)
        return {"label": result[0], "probability": result[1].tolist()}





