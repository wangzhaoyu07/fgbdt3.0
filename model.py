from datetime import datetime
import os
import shutil
import unittest
import torch
import numpy as np
import pickle
from worker import Worker
from server import ParameterServer
from sklearn.metrics import accuracy_score

os.chdir(os.path.join(__file__,os.path.pardir))


class FedAveragingGradsTestSuit(unittest.TestCase):
    RESULT_DIR = 'result'
    N_VALIDATION = 10000
    TEST_BASE_DIR = 'tmp/'

    def setUp(self):
        self.seed = 0
        self.use_cuda = False
        self.batch_size = 1
        self.test_batch_size = 100 #1000
        self.lr = 0.001
        self.n_max_rounds = 100
        self.log_interval = 20
        self.n_round_samples = 1600
        self.testbase = self.TEST_BASE_DIR
        self.n_users = 5

        self.feature_num = 561
        self.boosting_round = 1
        self.booster_dim = 6  # 原来的dim是14，对应14种标签
        self.bin_num = 16
        self.learning_rate = 1
        self.max_depth = 5
        self.testworkdir = os.path.join(self.testbase, 'competetion-test')
        self.modeldir = os.path.join(self.testbase, 'model')
        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)
            
        if not os.path.exists(self.modeldir):
            os.makedirs(self.modeldir)
            
        self.ps = ParameterServer(self.feature_num, self.boosting_round, self.booster_dim, self.bin_num,
                                  self.learning_rate, self.max_depth, self.testworkdir, self.RESULT_DIR, self.modeldir)

        self.workers = []
        print('cpu count: ', os.cpu_count())
        for u in range(0, self.n_users):
            #  for u in range(0, self.n_users):
            self.workers.append(Worker(self.booster_dim, self.bin_num, self.feature_num, u))

    def _clear(self):
        shutil.rmtree(self.testworkdir)
        shutil.rmtree(self.modeldir)

    def tearDown(self):
            self._clear()

    def test_federated_averaging(self):
        self.ps.build(self.workers)
        self.ps.ensemble()
        self.ps.save_testdata_prediction()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(FedAveragingGradsTestSuit('test_federated_averaging'))
    return suite


def main():
    runner = unittest.TextTestRunner()
    runner.run(suite())


def get_predict():
    ret=[]
    f=open('result/result.txt','r',encoding='utf-8')
    tp=f.readline()
    while tp:
        try:
            tp=int(tp[0])
            ret.append(tp)
        except:
            pass
        tp=f.readline()
    return ret

if __name__ == '__main__':
    print('cpu count: ', os.cpu_count())
    main()

    #计算得分
    with open('test/testing-X.pkl', 'rb') as fin:
        data = pickle.load(fin)

    label=data['y']
    predict=get_predict()
    print(accuracy_score(label,predict))