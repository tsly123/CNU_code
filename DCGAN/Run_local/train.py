"""
tsly, Sat Dec  8 16:34:49 2018

This code is taken from https://github.com/tatsy/keras-generative with small modifications.

"""

import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')

from models import DCGAN
import celebA_32
 
models = {
    'dcgan': DCGAN
}

#%%
def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Training DCGAN for CelebA')
    parser.add_argument('--model', type=str, default='dcgan')
    parser.add_argument('--dataset', type=str, required=False)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--output', default='output')
    parser.add_argument('--zdims', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--testmode', action='store_true')

    args = parser.parse_args()

    # select gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Make output direcotiry if not exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)


    datasets = celebA_32.load_data()

    # Construct model
    if args.model not in models:
        raise Exception('Unknown model:', args.model)

    model = models[args.model](
        input_shape=datasets.shape[1:],
        z_dims=args.zdims,
        output=args.output
    )

    if args.testmode:
        model.test_mode = True

    if args.resume is not None:
        model.load_model(args.resume)

    # Training loop
    datasets = datasets.images * 2.0 - 1.0
    samples = np.random.normal(size=(100, args.zdims)).astype(np.float32)
    model.main_loop(datasets, samples,
        epochs=args.epoch,
        batchsize=args.batchsize,
        reporter=['loss', 'g_loss', 'd_loss', 'g_acc', 'd_acc'])

if __name__ == '__main__':
    main()
