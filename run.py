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
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras import datasets, utils, backend
backend.set_session(session)
import numpy as np
import time
import pickle
import logging
logging.basicConfig(filename=f'./reports/experiment_logs/{EXPERIMENT_NAME}.log',level=logging.DEBUG)


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

    pd.DataFrame(history_child.history).to_csv(f"reports/experiment_logs/{EXPERIMENT_NAME}.epoch{epoch}.csv")
    val_acc_history = history_child.history["val_acc"]
    max_val_acc = round(max(val_acc_history), 3)

    logging.info('-> Child max validation accuracy: %.3f (elaspsed time: %ds)' % (max_val_acc, (toc-tic)))
    mem_accuracies.append(max_val_acc)

    if len(mem_softmaxes) > 5:
        # maybe better to let some epochs pass, so that the normalization is more robust
        controller.fit(mem_softmaxes, mem_accuracies)
    logging.info("-")

    if epoch%REPORT_PERIOD==0:
        logging.info ("Writing periodic report ...")
        epoch_best_policies = []
        for i, subpolicy in enumerate(subpolicies):
            epoch_best_policies.append(str(subpolicy))

        best_policy_report[epoch] = {
            "best_policies" : epoch_best_policies,
            "best_validation_accuracy" : max_val_acc
        }

        max_val_acc_str = str(max_val_acc).replace("0.","")
        report_file_path = f"./reports/best_policies/experiment{EXPERIMENT_NAME}.epoch{epoch}.max_val_acc_{max_val_acc_str}.best_policies.pkl"
        with open(report_file_path, 'wb') as f:
            pickle.dump(best_policy_report, f)

        controller.model.save_weights(
            f"./reports/controller_weights/experiment{EXPERIMENT_NAME}.epoch{epoch}.max_val_acc_{max_val_acc_str}.controller_weights.h5",
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
