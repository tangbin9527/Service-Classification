import pickle
import csv
import h5py
import pandas as pd
import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import backend as K
import config
from model import COBERT

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
                    level=logging.DEBUG,                                                # define the level of log
                    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',    # define the format of log
                    datefmt='%Y-%m-%d %A %H:%M:%S',                                     # define the famete of time
                    )

import os
gpus = tf.config.list_physical_devices(device_type='GPU')
# print(gpus)
tf.config.set_visible_devices(devices=gpus[0], device_type='GPU')
# pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com tensorflow-gpu==2.2.0

class ServiceClassification:
    def __init__(self, args=None):
        self.args = args

        self.path = self.args.data_path
        self.train_data_path = self.path + self.args.train_data_path
        self.test_data_path = self.path + self.args.test_data_path
        self.model_path = self.path + self.args.model_name
        
        self.epoch = self.args.epoch

    def train_generator(self, traindata):
        while True: 
            for d in traindata:
                x_train = d[0]
                y_train = d[1]
                yield x_train, y_train

    def test_generator(self, testdata):
        while True: 
            for d in testdata:
                x_test = d[0]
                y_test = d[1]
                yield x_test, y_test

    def load_train_data(self):
        """Load Train Data"""
        f = open(self.train_data_path, 'rb')
        traindata = pickle.load(f)
        f.close()
        tranning_steps_per_epoch = len(traindata)
        return traindata, tranning_steps_per_epoch

    def load_test_data(self):
        f = open(self.test_data_path, 'rb')
        testdata = pickle.load(f)
        f.close()
        validation_steps = len(testdata)
        return testdata, validation_steps

    def Precision(self, Y_test, predY_test):
        return metrics.precision_score(Y_test, predY_test, average='macro')

    def Recall(self, Y_test, predY_test):
        return metrics.recall_score(Y_test, predY_test, average='macro')

    def F_score(self, Y_test, predY_test):
        return metrics.f1_score(Y_test, predY_test, labels=range(0, 50), average='macro')

    def predictTestData(self, testdata, model):
        predY_test = []
        Y_test = []
        for d in testdata:
            predY_test.append(model.predict(d[0]))
            Y_test.append(d[1])

        predY_test = np.concatenate(predY_test, axis=0)
        Y_test = np.concatenate(Y_test, axis=0)

        test_label = []
        for i in range(Y_test.shape[0]):
            for j in range(len(Y_test[i])):
                if Y_test[i][j] == 1:
                    test_label.append(j)
        return predY_test, test_label
    
    def train(self, model):
        best_top5 = 0
        best_top1 = 0
        best_p = 0
        best_r = 0
        best_f1 = 0

        best_epoch = 0
        best_model = 0

        best_list = []

        top5_list = []
        top1_list =[]
        p_list = []
        r_list = []
        f_list = []

        traindata, tranning_steps_per_epoch = self.load_train_data()
        testdata, validation_steps = self.load_test_data()
        for i in range(self.epoch):
            history = model.fit(self.train_generator(traindata), tranning_steps_per_epoch)
            loss_test, top5error_test, top1error_test = model.evaluate(self.test_generator(testdata), steps=validation_steps)
            logger.info("current epoch = " + str(i+1))
            logger.info("top1 accuracy = " + str(top1error_test))
            logger.info("top5 accuracy = " + str(top5error_test))

            # predict the result
            predY_test, test_label = self.predictTestData(testdata, model)
            non_onehot_pred_test = np.argmax(predY_test, axis=1)

            precision = self.Precision(test_label, non_onehot_pred_test)
            recall = self.Recall(test_label, non_onehot_pred_test)
            f1 = self.F_score(test_label, non_onehot_pred_test)
            logger.info("precision = " + str(precision))
            logger.info("recall = " + str(recall))
            logger.info("f1 = " + str(f1))

            top5_list.append(top5error_test)
            top1_list.append(top1error_test)
            p_list.append(precision)
            r_list.append(recall)
            f_list.append(f1)

            if f1 > best_f1:
                best_top5 = top5error_test
                best_top1 = top1error_test
                best_p = precision
                best_r = recall
                best_f1 = f1
                best_epoch = i+1
                model.save(self.model_path)

            logger.info("best epoch = {} best f1 = {}".format(best_epoch, best_f1))
            print(metrics.classification_report(test_label, non_onehot_pred_test, labels=range(0, 50), output_dict=True))
        best_list.append([best_top5, best_top1, best_p, best_r, best_f1])
        
        logger.info(top5_list)
        logger.info(top1_list)
        logger.info(p_list)
        logger.info(r_list)
        logger.info(f_list)
        logger.info(best_list)

    def eval(self, model):
        testdata, validation_steps = self.load_test_data()
        model.load(self.model_path)

        loss_test, top5error_test, top1error_test = model.evaluate(self.test_generator(testdata), steps=validation_steps)
        logger.info("top1 accuracy = " + str(top1error_test))
        logger.info("top5 accuracy = " + str(top5error_test))

        predY_test, test_label = self.predictTestData(testdata, model)
        non_onehot_pred_test = np.argmax(predY_test, axis=1)

        precision = self.Precision(test_label, non_onehot_pred_test)
        recall = self.Recall(test_label, non_onehot_pred_test)
        f1 = self.F_score(test_label, non_onehot_pred_test)
        logger.info("precision = " + str(precision))
        logger.info("recall = " + str(recall))
        logger.info("f1 = " + str(f1))

        print(metrics.classification_report(test_label, non_onehot_pred_test, labels=range(0, 50), output_dict=True))


def main():
    args = config.parse_args()
    # print(args.data_path)
    # print(args.data_path + args.train_data_path)

    classifier = ServiceClassification(args)

    logger.info('Build Model')
    model = COBERT(args)
    model.build()
    optimizer = "sgd"
    model.compile(optimizer=optimizer)

    if args.mode == "train":
        classifier.train(model)
    
    if args.mode == "eval":
        print("eval model")
        classifier.eval(model)


if __name__ == "__main__":
    main()
