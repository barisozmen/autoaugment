# (C) 2018 Ricardo Cruz <ricardo.pdm.cruz@gmail.com>

import argparse, sys
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('controller_epochs', type=int)
parser.add_argument('--reduced', action='store_true')
parser.add_argument('--child-epochs', default=120, type=int)
parser.add_argument('--child-batch-size', default=128, type=int)
args = parser.parse_args()

# silence tensorflow annoying logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tell tensorflow to not use all resources
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras import datasets, utils, backend
backend.set_session(session)
import numpy as np
import time

# datasets in the AutoAugment paper:
# CIFAR-10, CIFAR-100, SVHN, and ImageNet
# SVHN = http://ufldl.stanford.edu/housenumbers/

if hasattr(datasets, args.dataset):
    (Xtr, ytr), (Xts, yts) = getattr(datasets, args.dataset).load_data()
else:
    sys.exit('Unknown dataset %s' % dataset)
if args.reduced:
    ix = np.random.choice(len(Xtr), 4000, False)
    Xtr = Xtr[ix]
    ytr = ytr[ix]

# we don't normalize the data because that is done during data augmentation
ytr = utils.to_categorical(ytr)
yts = utils.to_categorical(yts)

# Experiment parameters

import mycontroller, mychild
controller = mycontroller.Controller()
mem_softmaxes = []
mem_accuracies = []

for epoch in range(args.controller_epochs):
    print('Controller: Epoch %d / %d' % (epoch+1, args.controller_epochs))

    softmaxes, subpolicies = controller.predict(mycontroller.SUBPOLICIES, Xtr)
    for i, subpolicy in enumerate(subpolicies):
        print('# Sub-policy %d' % (i+1))
        print(subpolicy)
    mem_softmaxes.append(softmaxes)

    print("Creating child model ...")
    child_model = mychild.create_simple_conv(Xtr.shape[1:])
    child = mychild.Child(child_model, args.child_batch_size, args.child_epochs)

    print("running mycontroller.autoaugment ...")
    tic = time.time()
    aug = mycontroller.autoaugment(subpolicies, Xtr, ytr, child.batch_size)
    print("fitting child model ...")
    child.fit(aug, len(Xtr) // child.batch_size)
    toc = time.time()

    accuracy = child.evaluate(Xts, yts)
    print('-> Child accuracy: %.3f (elaspsed time: %ds)' % (accuracy, (toc-tic)))
    mem_accuracies.append(accuracy)

    if len(mem_softmaxes) > 5:
        # maybe better to let some epochs pass, so that the normalization is more robust
        controller.fit(mem_softmaxes, mem_accuracies)
    print()

print()
print('Best policies found:')
print()
_, subpolicies = controller.predict(mycontroller.SUBPOLICIES, Xtr)
for i, subpolicy in enumerate(subpolicies):
    print('# Subpolicy %d' % (i+1))
    print(subpolicy)
