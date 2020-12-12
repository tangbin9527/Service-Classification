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
tf.config.set_visible_devices(devices=gpus[5], device_type='GPU')
# pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com tensorflow-gpu==2.2.0

class ServiceClassification:
    def __init__(self, args=None):
        self.args = args

        self.path = self.args.data_path
        self.train_data_path = self.path + self.args.train_data_path
        self.test_data_path = self.path + self.args.test_data_path
        self.model_name = self.args.model_name
        
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

    def save_model_epoch(self, model, epoch):
        if not os.path.exists(self.path+'models/' + self.model_name+'/'):
            os.makedirs(self.path+'models/' + self.model_name +'/')
        model.save("{}models/{}/epo{:d}.hdf5".format(self.path, self.model_name, epoch), overwrite=True)

    def load_model_epoch(self, model, epoch):
        assert os.path.exists(
            "{}models/{}/epo{:d}.hdf5".format(self.path, self.model_name, epoch)), "Weights at epoch {:d} not found".format(epoch)
        model.load("{}models/{}/epo{:d}.hdf5".format(self.path, self.model_name, epoch))

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
        save_every = self.args.save_every
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

            print(metrics.classification_report(test_label, non_onehot_pred_test, labels=range(0, 50), output_dict=True))
            if save_every is not None and (i+1) % save_every == 0:
                self.save_model_epoch(model, i+1)
                # model.save(self.model_path)
        

    def eval(self, model):
        testdata, validation_steps = self.load_test_data()
        # model.load(self.model_path)

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
        if args.reload > 0:
            classifier.load_model_epoch(model, args.reload)
        classifier.eval(model)


if __name__ == "__main__":
    main()
