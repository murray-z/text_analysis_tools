# -*- coding: utf-8 -*-

import os
import sys
import fasttext.FastText as ff


class FastText:
    def __init__(self, save_model_path=None, train_data_path=None, test_data_path=None, type="train",
                 k=1, threshold=0.0):
        """
        fasttext 文本分类
        :param train_data_path: 训练文本路径
        :param save_model_path: 模型保存路径
        :param test_data_path: 测试文本路径， 默认为 None
        :param type: 运行模式，“train/prediict”
        """
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.save_model_path = save_model_path
        self.type = type
        self.k = k
        self.threshold = threshold
        if self.type == "predict":
            if not os.path.exists(self.save_model_path):
                print("MODEL: {} is not EXIST ....")
                sys.exit()
            print("LOAD MODEL FROM: {}".format(self.save_model_path))
            self.classifier = ff.load_model(self.save_model_path)
        else:
            self.classifier = None

    def train(self):
        classifier = ff.train_supervised(self.train_data_path, label_prefix="__label__")
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





