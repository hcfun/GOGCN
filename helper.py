import sys, os, random, json, uuid, time, argparse, logging, logging.config
import numpy as np
from random import randint
from collections import defaultdict as ddict, Counter
from orderedset import OrderedSet
from pprint import pprint

# PyTorch related imports
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn import Parameter as Param
from torch.utils.data import DataLoader
from torch_scatter import scatter_add

np.set_printoptions(precision=4)


def set_gpu(gpus):
    """
    Sets the GPU to be used for the run

    Parameters
    ----------
    gpus:           List of GPUs to be used for the run

    Returns
    -------

    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def get_logger(name, log_dir, config_dir):
    """
    Creates a logger object

    Parameters
    ----------
    name:           Name of the logger file
    log_dir:        Directory where logger file needs to be stored
    config_dir:     Directory from where log_config.json needs to be read

    Returns
    -------
    A logger object which writes to both file and stdout

    """
    config_dict = json.load(open(config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


def get_combined_results(left_results, right_results):
    """
    Computes the average based on head and tail prediction results

    Parameters
    ----------
    left_results:   results of predicting the tail according to (sub, rel)
    right_results: 	results of predicting the head according to (obj, rel_inverse)

    Returns
    -------
    Average prediction results

    """

    results = {}
    count = float(left_results['count'])

    results['left_mr'] = round(left_results['mr'] / count, 5)  # 保留五位小数 得到最终的mr结果
    results['left_mrr'] = round(left_results['mrr'] / count, 5)  # 保留五位小数 得到最终的mrr结果
    results['right_mr'] = round(right_results['mr'] / count, 5)  # 保留五位小数
    results['right_mrr'] = round(right_results['mrr'] / count, 5)
    results['mr'] = round((left_results['mr'] + right_results['mr']) / (2 * count), 5)  # 左右结合 mr
    results['mrr'] = round((left_results['mrr'] + right_results['mrr']) / (2 * count), 5)  # 左右结合 mrr

    for k in range(10):
        results['left_hits@{}'.format(k + 1)] = round(left_results['hits@{}'.format(k + 1)] / count, 5)  # 保留五位小数 得到最终的hit结果
        results['right_hits@{}'.format(k + 1)] = round(right_results['hits@{}'.format(k + 1)] / count, 5)
        results['hits@{}'.format(k + 1)] = round((left_results['hits@{}'.format(k + 1)] + right_results['hits@{}'.format(k + 1)]) / (2 * count), 5)  # 左右结合 hit
    return results


def get_param(shape):
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param


def com_mult(a, b):
	r1, i1 = a[..., 0], a[..., 1] #最后一维上第一组元素、第二组元素
	r2, i2 = b[..., 0], b[..., 1] #最后一维上第一组元素、第二组元素
	return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)


def conj(a):
	a[..., 1] = -a[..., 1] # 最后一维第二组元素的相反数
	return a

def cconv(a, b):
	return torch.irfft(com_mult(torch.rfft(a, 1), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def ccorr(a, b):
	#torch.irfft：从复到实的反离散傅里叶变换

	#torch.rfft:从实到复的离散傅里叶变换
	return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))