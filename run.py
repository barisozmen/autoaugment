# (C) 2018 Ricardo Cruz <ricardo.pdm.cruz@gmail.com>

import argparse, sys
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('controller_epochs', type=int)
parser.add_argument('--reduce-to', default=4000, type=int)
parser.add_argument('--child-epochs', default=120, type=int)
parser.add_argument('--child-batch-size', default=128, type=int)
parser.add_argument('--report-period', default=20, type=int)
args = parser.parse_args()

import datetime
now = datetime.datetime.now()
EXPERIMENT_NAME = f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}"
# give best policies report for each REPORT_PERIOD epochs of the controller
REPORT_PERIOD = args.report_period
best_policy_report = {}
VALIDATION_SET_SIZE = 1000


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # silence tensorflow annoying logs
import pandas as pd

import tensorflow as tf
# tell tensorflow to not use all resources
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
session = tf.Session() # session = tf.Session(config=config)

from keras import datasets, utils, backend
backend.set_session(session)
import numpy as np
import time
import pickle
import logging
if not os.path.exists("./reports/"):
    os.mkdir("./reports")
if not os.path.exists("./reports/experiments"):
    os.mkdir("./reports/experiments")
EXPERIMENT_LOG_FOLDER = f"reports/experiments/{EXPERIMENT_NAME}"
os.mkdir(EXPERIMENT_LOG_FOLDER)
logging.basicConfig(filename=f'./{EXPERIMENT_LOG_FOLDER}/info.log',level=logging.DEBUG)


# datasets in the AutoAugment paper:
# CIFAR-10, CIFAR-100, SVHN, and ImageNet
# SVHN = http://ufldl.stanford.edu/housenumbers/

if hasattr(datasets, args.dataset):
    (Xtr, ytr), (Xts, yts) = getattr(datasets, args.dataset).load_data()
else:
    sys.exit('Unknown dataset %s' % dataset)

# reduce training dataset
ix = np.random.choice(len(Xtr), args.reduce_to, False)
Xtr = Xtr[ix]
ytr = ytr[ix]

# reduce validation dataset
iy = np.random.choice(len(Xts), VALIDATION_SET_SIZE, False)
Xts = Xts[iy]
yts = yts[iy]

# we don't normalize the data because that is done during data augmentation
ytr = utils.to_categorical(ytr)
yts = utils.to_categorical(yts)


# Experiment parameters
import mycontroller, mychild
controller = mycontroller.Controller()
mem_softmaxes = []
mem_accuracies = []

for epoch in range(args.controller_epochs):
    logging.info('Controller: Epoch %d / %d' % (epoch+1, args.controller_epochs))

    softmaxes, subpolicies = controller.predict(mycontroller.SUBPOLICIES, Xtr)
    for i, subpolicy in enumerate(subpolicies):
        logging.info('# Sub-policy %d' % (i+1))
        logging.info(subpolicy)
    mem_softmaxes.append(softmaxes)

    logging.info("Creating child model ...")
    child_model = mychild.create_simple_conv(Xtr.shape[1:])
    child = mychild.Child(child_model, args.child_batch_size, args.child_epochs)

    logging.info("running mycontroller.autoaugment ...")
    tic = time.time()
    aug = mycontroller.autoaugment(subpolicies, Xtr, ytr, child.batch_size)
    logging.info("fitting child model ...")
    history_child = child.fit(aug, len(Xtr) // child.batch_size, Xts, yts)
    toc = time.time()

    history_df = pd.DataFrame(history_child.history)
    history_df = history_df.round(3)

    # report child model history
    pd.DataFrame(history_df).to_csv(f"./{EXPERIMENT_LOG_FOLDER}/epoch{epoch}.child_history.csv")
    max_val_acc = history_df["val_acc"].max()

    logging.info('-> Child max validation accuracy: %.3f (elaspsed time: %ds)' % (max_val_acc, (toc-tic)))
    mem_accuracies.append(max_val_acc)

    if len(mem_softmaxes) > 5:
        # maybe better to let some epochs pass, so that the normalization is more robust
        controller.fit(mem_softmaxes, mem_accuracies)
    logging.info("-")

    # report best policies of the epoch
    max_val_acc_str = str(max_val_acc).replace("0.", "")
    epoch_best_policies = []
    for i, subpolicy in enumerate(subpolicies):
        epoch_best_policies.append(str(subpolicy))
    report_file_path = f"./{EXPERIMENT_LOG_FOLDER}/epoch{epoch}.max_val_acc_{max_val_acc_str}.best_policies.csv"
    pd.DataFrame(epoch_best_policies).to_csv(report_file_path, index=False)


    if epoch%REPORT_PERIOD==0:
        controller.model.save_weights(
            f"./{EXPERIMENT_LOG_FOLDER}/epoch{epoch}.controller_weights.h5",
            overwrite=True
        )

logging.info("-")
logging.info('Best policies found:')
logging.info("-")
_, subpolicies = controller.predict(mycontroller.SUBPOLICIES, Xtr)
for i, subpolicy in enumerate(subpolicies):
    logging.info('# Subpolicy %d' % (i+1))
    logging.info(subpolicy)
    print('# Subpolicy %d' % (i + 1))
    print(subpolicy)
