#!/usr/bin/env python
#-----------------------------------------------------------------------------
# File Name : thom_gesture.py
# Author: Tielin Zhang in CASIA
#-----------------------------------------------------------------------------
# from dcll.pytorch_libdcll import *
# from dcll.experiment_tools import *
from dcll.load_dvsgestures_sparse import *
# import argparse
from tqdm import tqdm
# import pickle
import ipdb


batch_size = 72
n_iters = 500
n_iters_test = 1800
in_channels = 2
ds = 4
im_dims = im_width, im_height = (128//ds, 128//ds)
dt = 1000 #us
valid = False
test_offset = 0
#Load data
gen_train, _ = create_data(
        batch_size = batch_size,
        chunk_size = n_iters,
        size = [in_channels, im_width, im_height],
        ds = ds,
        dt = dt)


_, gen_test = create_data(
        batch_size = batch_size,
        chunk_size = n_iters_test,
        size = [in_channels, im_width, im_height],
        ds = ds,
        dt = dt)

# def generate_test(gen_test, n_test:int, offset=0):
#     input_test, labels_test = gen_test.next(offset=offset)
#     input_tests = []
#     labels1h_tests = []
#     n_test = min(n_test,int(np.ceil(input_test.shape[0]/batch_size)))
#     for i in range(n_test):
#         input_tests.append( torch.Tensor(input_test.swapaxes(0,1))[:,i*batch_size:(i+1)*batch_size].reshape(n_iters_test,-1,in_channels,im_width,im_height))
#         labels1h_tests.append(torch.Tensor(labels_test[:,i*batch_size:(i+1)*batch_size]))
#     return n_test, input_tests, labels1h_tests


# n_test, input_tests, labels1h_tests = generate_test(gen_test, n_test=1 if valid else 100, offset = test_offset)

# ipdb.set_trace()

inputdata, labels = gen_train.next()
print('inputdata.shape = ',str(inputdata.shape))
print('labels.shape = ', str(labels.shape))
print('finish')

inputdata, labels = gen_test.next()
print('inputdata.shape = ',str(inputdata.shape))
print('labels.shape = ', str(labels.shape))
print('finish')

