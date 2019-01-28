# (C) 2018 Baris Ozmen <hbaristr@gmail.com>

import argparse, sys
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--opt-iterations', default=1000, type=int)
parser.add_argument('--opt-random-states', default=5, type=int)
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


# SMAC3 imports
# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

# Experiment parameters
import mychild
from augmenter import Augmenter


def save_child_model_history(history_child):
    history_df = pd.DataFrame(history_child.history)
    history_df = history_df.round(3)

    pd.DataFrame(history_df).to_csv(f"./{EXPERIMENT_LOG_FOLDER}/epoch{epoch}.child_history.csv")
    max_val_acc = history_df["val_acc"].max()

    return max_val_acc

augmenter = Augmenter()

def objective(params):

    Xtr_aug, ytr_aug = augmenter.run(Xtr, ytr, aug_parameters=params)

    child_model = mychild.create_simple_conv(Xtr_aug.shape[1:])
    child = mychild.Child(child_model, args.child_batch_size, args.child_epochs)

    logging.info("fitting child model ...")
    child_history = child.fit(Xtr_aug, ytr_aug, len(Xtr_aug) // child.batch_size, Xts, yts)

    max_val_acc = save_child_model_history(child_history)
    logging.info(f'-> Child max validation accuracy: {max_val_acc}')

    # the lower the better
    return 1 - max_val_acc



# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()
# We define a few possible types of SVM-kernels and add them as "kernel" to our cs
transform_type = CategoricalHyperparameter("transform_type", ["ShearX", "ShearY", "TranslateX", "TranslateY"], default_value="ShearX")
cs.add_hyperparameter(transform_type)
# There are some hyperparameters shared by all kernels
magnitude = UniformFloatHyperparameter("magnitude", -1.0, 1.0, default_value=0.0)
cs.add_hyperparameter(magnitude)

# Scenario object
scenario = Scenario({
    "run_obj": "quality",                   # we optimize quality (alternatively runtime)
    "runcount-limit": args.opt_iterations,   # maximum function evaluations
    "cs": cs,                               # configuration space
    "deterministic": "true",
    "abort_on_first_run_crach": "false"
})

# Optimize, using a SMAC-object
print("Optimizing...")
smac = SMAC(
    scenario=scenario,
    rng=np.random.RandomState(10),
    tae_runner=objective
)
incumbent = smac.optimize()

inc_value = objective(incumbent)

print("Optimized Value: %.2f" % (inc_value))

