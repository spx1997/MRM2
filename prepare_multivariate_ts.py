from scipy.io import arff
import os
import numpy as np
import time
import pandas as pd
from sklearn import preprocessing


def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a


flist = ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories', 'Cricket',
         'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'EthanolConcentration', 'ERing', 'FaceDetection', 'FingerMovements',
         'HandMovementDirection', 'Handwriting', 'Heartbeat', 'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST',
         'MotorImagery', 'NATOPS', 'PenDigits', 'PEMS-SF', 'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1',
         'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']


for fname in flist:
    print('deal: ', fname)
    mode = 'TRAIN'
    train_data = arff.loadarff('./data/Multivariate_arff/{}/{}_{}.arff'.format(fname, fname, mode))
    train_df = pd.DataFrame(train_data[0])
    train_data_list = []
    train_df.columns = ['attribute', 'label']
    for i in range(len(train_df)):
        t = [list(train_df['attribute'][i][c]) for c in range(len(train_df['attribute'][i]))]
        train_data_list.append(t)
    train_data_np = np.array(train_data_list)
    train_data_np = set_nan_to_zero(train_data_np)
    if not os.path.exists('./data/Multivariate_ts_np/{}'.format(fname)):
        os.makedirs('./data/Multivariate_ts_np/{}'.format(fname))
    np.save('./data/Multivariate_ts_np/{}/{}_{}.npy'.format(fname, fname, mode), train_data_np)
    train_y = np.array(train_df['label'].tolist()).astype(np.str)
    le = preprocessing.LabelEncoder()
    le.fit(train_y)
    train_y = le.transform(train_y)
    np.save('./data/Multivariate_ts_np/{}/{}_{}_label.npy'.format(fname, fname, mode), train_y)

    mode = 'TEST'
    test_data = arff.loadarff('./data/Multivariate_arff/{}/{}_{}.arff'.format(fname, fname, mode))
    test_df = pd.DataFrame(test_data[0])
    test_data_list = []
    test_df.columns = ['attribute', 'label']
    for i in range(len(test_df)):
        t = [list(test_df['attribute'][i][c]) for c in range(len(test_df['attribute'][i]))]
        test_data_list.append(t)
    test_data_np = np.array(test_data_list)
    test_data_np = set_nan_to_zero(test_data_np)
    if not os.path.exists('./data/Multivariate_ts_np/{}'.format(fname)):
        os.makedirs('./data/Multivariate_ts_np/{}'.format(fname))
    np.save('./data/Multivariate_ts_np/{}/{}_{}.npy'.format(fname, fname, mode), test_data_np)
    test_y = np.array(test_df['label'].tolist()).astype(np.str)
    test_y = le.transform(test_y)
    np.save('./data/Multivariate_ts_np/{}/{}_{}_label.npy'.format(fname, fname, mode), test_y)