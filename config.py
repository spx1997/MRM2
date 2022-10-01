import argparse
import time


class TSCConfig(object):
    def __init__(self):
        # config for training
        self.data_dir = './data'
        self.out_dir = './out_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.lr = 0.001
        self.max_epoch = 500
        self.batch_size = 128
        self.nclass = None
        self.gpu_nums = 1
        self.start_cuda_id = 0
        self.weight_decay = 0.01
        self.pretrain_model_dir = ''
        self.start_dataset_id = 0
        self.lambda_loss = 0.5
        # self.gamma_loss = 0.5
        self.lr_restart = 0.001
        self.train_from_scratch_max_epoch = 400
        self.model_name = ''
        self.alpha = 1.
        self.beta = 1.
        self.repeat = 0

    def to_dict(self):
        return self.__dict__

    def update(self, dic):
        self.__dict__.update(dic)


def set_parser():
    config = TSCConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/Multivariate_ts_np',
                        type=str, required=False, help='the directory of UEA')
    parser.add_argument('--out_dir', default=None, type=str, required=False,
                        help='The output directory where model predictions and checkpoints will be written. ')
    parser.add_argument('--max_epoch', default=500, type=int, required=False,
                        help='the max epoch for training the model')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate for the model')
    parser.add_argument('--lr_restart', default=0.001, type=float,
                        help='')
    parser.add_argument('--gpu_nums', default=1, type=int,
                        help='the number of the gpu')
    parser.add_argument('--start_cuda_id', default=0, type=int,
                        help="if use gpu, start_cuda_id set the start gpu device id")
    parser.add_argument('--batch_size', default=128, type=int,
                        help='the number sample of a batch')
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help="")
    parser.add_argument('--pretrain_model_dir', default='', type=str,
                        help='the directory for the pretrain model')
    parser.add_argument('--start_dataset_id', default=0, type=int,
                        help='')
    parser.add_argument('--lambda_loss', default=0.01, type=float,
                        help="")
    parser.add_argument('--model_name', default='', type=str,
                        help="")
    parser.add_argument('--train_from_scratch_max_epoch', default=400, type=int,
                        help="")
    parser.add_argument('--repeat', default=0, type=int,
                        help="")
    args = parser.parse_args()
    config.data_dir = args.data_dir
    config.out_dir = args.out_dir
    config.lr = args.lr
    config.max_epoch = args.max_epoch
    config.batch_size = args.batch_size
    config.gpu_nums = args.gpu_nums
    config.start_cuda_id = args.start_cuda_id

    config.weight_decay = args.weight_decay
    config.pretrain_model_dir = args.pretrain_model_dir
    config.start_dataset_id = args.start_dataset_id
    config.lambda_loss = args.lambda_loss
    # config.gamma_loss = args.gamma_loss
    config.model_name = args.model_name
    config.lr_restart = args.lr_restart

    config.train_from_scratch_max_epoch = args.train_from_scratch_max_epoch
    config.repeat = args.repeat
    return config


multiflist = ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories', 'Cricket',
              'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'EthanolConcentration', 'ERing', 'FaceDetection',
              'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat', 'InsectWingbeat',
              'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery', 'NATOPS', 'PenDigits', 'PEMS-SF', 'PhonemeSpectra',
              'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump',
              'UWaveGestureLibrary']

batch_size = 512

mts_batchsize_config = \
    {'ArticularyWordRecognition': {
        "fcn": [64, 300],
        "inceptiontime": [64, 0],
        "resnet": [64, 300],
        "mlstm_fcn": [275, 150],
        "os_cnn": [275, 150],
    },
        'AtrialFibrillation': [1, 1],
        'BasicMotions': [10, 40],
        'CharacterTrajectories': [1422, 718],
        'Cricket': [108, 36],
        'DuckDuckGeese': [30, 40],
        'EigenWorms': {
            "fcn": [100, 0],
            "inceptiontime": [112, 0],
            "resnet": [80, 0],
            "mlstm_fcn": [100, 0],
            "os_cnn": [80, 0],
        },
        'Epilepsy': [8, 32],
        'EthanolConcentration': {
            "fcn": [4, 0],
            "inceptiontime": [4, 0],
            "resnet": [4, 0],
            "mlstm_fcn": [4, 0],
            "os_cnn": [4, 0],
        },
        'ERing': [20, 80],
        'FaceDetection': {
            "fcn": [8, 0],
            "inceptiontime": [1000, 500],
            "resnet": [16, 0],
            "mlstm_fcn": [32, 0],
            "os_cnn": [32, 0],
        },
        'FingerMovements': [80, 50],
        'HandMovementDirection': [100, 10],
        'Handwriting': [150, 10],
        'Heartbeat': [204, 102],
        'InsectWingbeat': [768, 768],
        'JapaneseVowels': [270, 300],
        'Libras': [180, 90],
        'LSST': {
            "fcn": [2048, 0],
            "inceptiontime": [512, 8],
            "resnet": [512, 8],
            "mlstm_fcn": [2459, 16],
            "os_cnn": [2459, 16],
        },
        'MotorImagery': [20, 80],
        'NATOPS': [5, 160],
        'PenDigits': [2048, 1024],
        'PEMS-SF': {
            "fcn": [5, 170],
            "inceptiontime": [6, 128],
            "resnet": [4, 0],
            "mlstm_fcn": [12, 128],
            "os_cnn": [5, 170],
        },
        'PhonemeSpectra': {
            "fcn": [1024, 0],
            "inceptiontime": [1658, 1024],
            "resnet": [1024, 0],
            "mlstm_fcn": [1024, 0],
            "os_cnn": [1024, 0],
        },
        'RacketSports': [120, 60],
        'SelfRegulationSCP1': {
            "fcn": [268, 80],
            "inceptiontime": [268, 80],
            "resnet": [128, 256, 0.1, 1.0],
            "mlstm_fcn": [268, 128, 1, 1.0],
            "os_cnn": [128, 256, 0.1, 1.0],
        },
        'SelfRegulationSCP2': [200, 100],
        'SpokenArabicDigits': [2048, 1024],
        'StandWalkJump': [9, 9],
        'UWaveGestureLibrary': [120, 200]}
