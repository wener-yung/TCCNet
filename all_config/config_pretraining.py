import argparse
import os
import os.path as osp

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0', help='train use gpu')
parser.add_argument('--video_batchsize', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=1e-4)
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
# scheduler
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--min_learning_rate', type=float, default=0.0000001)
parser.add_argument('--warmup_iteration', type=int, default=300)

# train schedule
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--nepochs', type=int, default=200)
parser.add_argument('--save_frequency', type=int, default=10)

# data
parser.add_argument('--video_dataset_list', type=str,
                    default=["CVC-ClinicDB-612", "CVC-ColonDB-300"])

parser.add_argument('--video_dataset_root', type=str,
                    default="dataset/TrainSet/")
parser.add_argument('--size', type=tuple, default=(352, 352))
parser.add_argument('--video_testset_root', type=str,
                    default="dataset/TestSet/")
parser.add_argument('--test_dataset_list', type=str,
                    default=["CVC-ColonDB-300", "CVC-ClinicDB-612-Valid", "CVC-ClinicDB-612-Test"])

parser.add_argument('--data_statistics', type=str, default="data/statistics.pth")
parser.add_argument('--mean_dic_path', type=str, default='data/mean.npy')
parser.add_argument('--sqc_path', type=str, default='data/sqc_pathlist.npy')

# model
parser.add_argument('--getall', type=bool, default=False)
parser.add_argument('--video_time_clips', type=int, default=15)
parser.add_argument('--memory_size', type=int, default=3)
parser.add_argument('--train_mode', type=str, default="pretraining",
                    help='pretraining | main_training')
parser.add_argument('--ifcut', type=bool, default=True)
parser.add_argument('--k_corrected', type=int, default=8)

# test
parser.add_argument('--load', type=str, default=None)

# name
parser.add_argument('--name', type=str, default="TCCNet")
parser.add_argument('--model', type=str, default='TCCNet')
parser.add_argument('--repo_name', type=str, default="model")
parser.add_argument('--setting', type=str, default="")

config = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

""" Data Dir  """
config.volna = 'TCCNet/'
config.snapshot_path = "saving/{}/{}/".format(config.name, config.repo_name)
config.writer_path = "saving/{}/log/".format(config.name)

config.visualize_path = config.volna + "saving/{}/{}/visual/".format(config.name, config.repo_name)
config.result_path = config.volna + "saving/{}/{}/result/".format(config.name, config.repo_name)
config.cvs_path = config.volna + "saving/"

# List[filepath] - Files to back up
config.backup = ['all_config/config_pretraining.py']
