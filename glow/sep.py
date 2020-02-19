#!/usr/bin/env python

import os,sys,time,signal,argparse

from itertools import permutations

import cv2
import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
from .utils import ResultLogger

from .model import prior,model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

learn = tf.contrib.learn

# Surpress verbose warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def generate_graph(data, axis_label, fname):
    plt.plot(data)
    plt.ylabel(axis_label)
    plt.savefig(fname)
    plt.close()

def get_inputs_no_split(hps, iterator1, iterator2, sess):
    """Gets two inputs without considering class"""
    if hps.direct_iterator:
        iterator1 = iterator1.get_next()
        gt0, y0 = sess.run(iterator1)
        iterator2 = iterator2.get_next()
        gt1, y1 = sess.run(iterator2)
    else:
        gt0, y0 = iterator1()
        gt1, y1 = iterator2()

    return gt0, gt1

def get_inputs_split(hps, iterator1, iterator2, sess, labels0, labels1):
    """
    Gets two inputs such that gt0 belongs to labels0 and
    gt1 belongs to labels1
    """
    gt0 = []
    gt1 = []

    # Continuously sample and assign to the right batch
    while ((len(gt0) < hps.n_batch_test) or (len(gt1) < hps.n_batch_test)):
        # Get a new set of inputs
        if hps.direct_iterator:
            iterator = iterator.get_next()
            x, y = sess.run(iterator)
        else:
            x, y = iterator()

        # Iterate and assign to the correct gt based on label
        for idx in range(y.shape[0]):
            if y[idx] in labels0 and (len(gt0) < hps.n_batch_test):
                gt0.append(x[idx])

            elif y[idx] in labels1 and (len(gt1) < hps.n_batch_test):
                gt1.append(x[idx])

    return np.stack(gt0, axis=0), np.stack(gt1, axis=0)


def generate_mixture(hps, iterator1, iterator2, sess):
    """
    Generates a mixture and returns the noisy samples used
    """
    gt0, gt1 = get_inputs_no_split(hps, iterator1, iterator2, sess)
    #gt0, gt1 = get_inputs_split(hps, iterator1, sess, range(5), range(5, 10))

    mixed = gt0/2. + gt1/2.

    # preprocessing
    mixed = mixed/255. - .5

    write_images(hps, mixed, 'mixed.png')

    # Scale gt to the same scale as x
    gt0 = gt0/255. - .5
    gt1 = gt1/255. - .5

    write_images(hps, gt0, 'xgt.png')
    write_images(hps, gt1, 'ygt.png')

    recon = (gt0 + gt1 - 2*mixed)**2

    # add some noise to make it interesting
    x0 = np.random.randn(*gt0.shape)
    x1 = np.random.randn(*gt1.shape)

    recon = (x0 + x1 - 2*mixed)**2

    write_images(hps, x0, 'x_init.png')
    write_images(hps, x1, 'y_init.png')

    return mixed, x0, x1, gt0, gt1


def infer(sess, model1, model2, hps, mixed, x0, x1, sigma, x_logprobs, y_logprobs, recons):
    # Example of using model in inference mode. Load saved model using hps.restore_path
    # Can provide x, y from files instead of dataset iterator
    # If model is uncondtional, always pass y = np.zeros([bs], dtype=np.int32)
    # tf.set_random_seed(7493220987)
    # np.random.seed(3344332)

    d = x0.shape[1]

    y = np.zeros([hps.n_batch_test], dtype=np.int32)

    eta = .00003 * (sigma / .01) ** 2
    lambda_recon = 1.0/(sigma**2)
    for i in range(100):
        grad_x0 = model1.grad_logprob(x0,y)
        grad_x1 = model2.grad_logprob(x1,y)

        epsilon0 = np.sqrt(2*eta)*np.random.randn(*x0.shape)
        epsilon1 = np.sqrt(2*eta)*np.random.randn(*x1.shape)

        x0 = x0 + eta * (grad_x0 - lambda_recon * (x0 + x1 - 2*mixed)) + epsilon0
        x1 = x1 + eta * (grad_x1 - lambda_recon * (x0 + x1 - 2*mixed)) + epsilon1

        if i == 99: # debugging
            recon = ((x0 + x1 - 2*mixed)**2).reshape(-1,3*d*d).sum(1).mean()
            normx = np.linalg.norm(grad_x0.reshape(-1,3*d*d),axis=1).mean()
            normy = np.linalg.norm(grad_x1.reshape(-1,3*d*d),axis=1).mean()
            logpx = model1.logprob(x0,y).mean()/(3*d*d*np.log(2))
            logpy = model2.logprob(x1,y).mean()/(3*d*d*np.log(2))
            print('recon: {}, logpx: {}, logpy: {}, normx: {}, normy: {}'.format(recon,logpx,logpy,normx,normy))


    # generate_graph(x_logprobs, "x logprob", os.path.join(hps.logdir, "x_logprob.png"))
    # generate_graph(y_logprobs, "y logprob", os.path.join(hps.logdir, "y_logprob.png"))
    # generate_graph(recons, "reconstruction error", os.path.join(hps.logdir, "recons.png"))

    return x0, x1


def write_images(hps, x, fname):
    n = hps.num_images
    d = x.shape[1]
    panel = np.zeros([n*d,n*d,3],dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            panel[i*d:(i+1)*d,j*d:(j+1)*d,:] = (255*(x[i*n+j]+.5)).clip(0,255).astype(np.uint8)[:,:,::-1]

    cv2.imwrite(os.path.join(hps.image_folder,'glow_' + hps.problem + '_' + fname), panel)


def estimate_nll(sess, model, hps, iterator):
    if hps.direct_iterator:
        iterator = iterator.get_next()

    print('Running inference on {} data points'.format(hps.full_test_its*hps.n_batch_test))
    logpz = []
    grad_logpz = []
    for it in range(hps.full_test_its):
        if hps.direct_iterator:
            # replace with x, y, attr if you're getting CelebA attributes, also modify get_data
            x, y = sess.run(iterator)
        else:
            x, y = iterator()

        # preprocess
        x = x/255. - .5
        x += np.random.randn(*(x.shape)) * hps.noise_level

        logpz.append(model.logprob(x,y))

    logpz = np.concatenate(logpz, axis=0)
    print('NLL = {}'.format(-logpz.mean()/(32*32*3*np.log(2))))


# ===
# Code for getting data
# ===
def get_data(hps, sess, category=None):
    if hps.image_size == -1:
        hps.image_size = {'mnist': 32, 'cifar10': 32, 'imagenet-oord': 64,
                          'imagenet': 256, 'celeba': 256, 'lsun_realnvp': 64, 'lsun': 256}[hps.problem]
    if hps.n_test == -1:
        hps.n_test = {'mnist': 10000, 'cifar10': 10000, 'imagenet-oord': 50000, 'imagenet': 50000,
                      'celeba': 3000, 'lsun_realnvp': 300*hvd.size(), 'lsun': 300*hvd.size()}[hps.problem]
    hps.n_y = {'mnist': 10, 'cifar10': 10, 'imagenet-oord': 1000,
               'imagenet': 1000, 'celeba': 1, 'lsun_realnvp': 1, 'lsun': 1}[hps.problem]

    hps.data_dir = {'mnist': None, 'cifar10': None, 'imagenet-oord': '/mnt/host/imagenet-oord-tfr', 'imagenet': '/mnt/host/imagenet-tfr',
                    'celeba': 'celeba-tfr', 'lsun_realnvp': 'lsun_realnvp', 'lsun': '/mnt/host/lsun'}[hps.problem]

    if hps.problem == 'lsun_realnvp':
        hps.rnd_crop = True
    else:
        hps.rnd_crop = False

    if category:
        hps.data_dir += ('/%s' % category)

    # Use anchor_size to rescale batch size based on image_size
    s = hps.anchor_size
    hps.local_batch_train = hps.n_batch_train * \
        s * s // (hps.image_size * hps.image_size)
    hps.local_batch_test = {64: 50, 32: 25, 16: 10, 8: 5, 4: 2, 2: 2, 1: 1}[
        hps.local_batch_train]  # round down to closest divisor of 50
    hps.local_batch_init = hps.n_batch_init * \
        s * s // (hps.image_size * hps.image_size)

    if hps.local_batch_test > 49:
        hps.num_images = 7
    elif hps.local_batch_test > 9:
        hps.num_images = 3

    print("Rank {} Batch sizes Train {} Test {} Init {}".format(
        hvd.rank(), hps.local_batch_train, hps.local_batch_test, hps.local_batch_init))

    if hps.problem in ['imagenet-oord', 'imagenet', 'celeba', 'lsun_realnvp', 'lsun']:
        hps.direct_iterator = True
        from .data_loaders import get_data as v
        train_iterator, test_iterator, data_init = \
            v.get_data(sess, hps.data_dir, hvd.size(), hvd.rank(), hps.pmap, hps.fmap, hps.local_batch_train,
                       hps.local_batch_test, hps.local_batch_init, hps.image_size, hps.rnd_crop)

    elif hps.problem in ['mnist', 'cifar10']:
        hps.direct_iterator = False
        from .data_loaders import get_mnist_cifar as v
        train_iterator, test_iterator, data_init = \
            v.get_data(hps.problem, hvd.size(), hvd.rank(), hps.dal, hps.local_batch_train,
                       hps.local_batch_test, hps.local_batch_init, hps.image_size)

    else:
        raise Exception()

    return train_iterator, test_iterator, data_init


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--restore_path", type=str, default='',
                        help="Location of checkpoint to restore")
    parser.add_argument("--logdir", type=str,
                        default='./logs', help="Location to save logs")
    parser.add_argument("--image_folder", type=str,
                        default='./output', help="Location to save image outputs")
    parser.add_argument("--noise_level", type=float, default=0,
                        help="Amount of noise to add")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Dataset hyperparams:
    parser.add_argument("--problem", type=str, default='cifar10',
                        help="Problem (mnist/cifar10/imagenet")
    parser.add_argument("--category1", type=str,
                        default='church_outdoor', help="LSUN category 1")
    parser.add_argument("--category2", type=str,
                        default='bedroom', help="LSUN category 2")
    parser.add_argument("--data_dir", type=str, default='',
                        help="Location of data")
    parser.add_argument("--dal", type=int, default=1,
                        help="Data augmentation level: 0=None, 1=Standard, 2=Extra")

    # New dataloader params
    parser.add_argument("--fmap", type=int, default=1,
                        help="# Threads for parallel file reading")
    parser.add_argument("--pmap", type=int, default=16,
                        help="# Threads for parallel map")

    # Optimization hyperparams:
    parser.add_argument("--n_train", type=int,
                        default=50000, help="Train epoch size")
    parser.add_argument("--n_test", type=int, default=-
                        1, help="Valid epoch size")
    parser.add_argument("--n_batch_train", type=int,
                        default=64, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=50, help="Minibatch size")
    parser.add_argument("--n_batch_init", type=int, default=256,
                        help="Minibatch size for data-dependent init")
    parser.add_argument("--optimizer", type=str,
                        default="adamax", help="adam or adamax")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--beta1", type=float, default=.9, help="Adam beta1")
    parser.add_argument("--polyak_epochs", type=float, default=1,
                        help="Nr of averaging epochs for Polyak and beta2")
    parser.add_argument("--weight_decay", type=float, default=1.,
                        help="Weight decay. Switched off by default.")
    parser.add_argument("--epochs", type=int, default=1000000,
                        help="Total number of training epochs")
    parser.add_argument("--epochs_warmup", type=int,
                        default=10, help="Warmup epochs")
    parser.add_argument("--epochs_full_valid", type=int,
                        default=50, help="Epochs between valid")
    parser.add_argument("--gradient_checkpointing", type=int,
                        default=1, help="Use memory saving gradients")

    # Model hyperparams:
    parser.add_argument("--image_size", type=int,
                        default=-1, help="Image size")
    parser.add_argument("--anchor_size", type=int, default=32,
                        help="Anchor size for deciding batch size")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=32,
                        help="Depth of network")
    parser.add_argument("--weight_y", type=float, default=0.00,
                        help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--n_bits_x", type=int, default=8,
                        help="Number of bits of x")
    parser.add_argument("--n_levels", type=int, default=3,
                        help="Number of levels")

    # Synthesis/Sampling hyperparameters:
    parser.add_argument("--n_sample", type=int, default=1,
                        help="minibatch size for sample")
    parser.add_argument("--epochs_full_sample", type=int,
                        default=50, help="Epochs between full scale sample")

    # Ablation
    parser.add_argument("--learntop", action="store_true",
                        help="Learn spatial prior")
    parser.add_argument("--ycond", action="store_true",
                        help="Use y conditioning")
    parser.add_argument("--flow_permutation", type=int, default=2,
                        help="Type of flow. 0=reverse (realnvp), 1=shuffle, 2=invconv (ours)")
    parser.add_argument("--flow_coupling", type=int, default=0,
                        help="Coupling type: 0=additive, 1=affine")

    return parser.parse_args()  # So error if typo


def main():

    # This enables a ctr-C without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    # Parse command-line arguments into hps
    hps = parse_args()

    # Initialize Horovod.
    hvd.init()

    # Create tensorflow session
    sess = tensorflow_session()

    # Download and load dataset.
    tf.set_random_seed(hvd.rank() + hvd.size() * hps.seed)
    np.random.seed(hvd.rank() + hvd.size() * hps.seed)

    # Hard-code some of the pre-trained model settings
    if hps.problem in ['mnist', 'cifar10']:
        hps.flow_coupling = 1    
    elif hps.problem == 'lsun_realnvp':
        hps.depth = 48
        hps.n_levels = 4
        hps.flow_coupling = 1
        hps.n_batch_test = 10

    # Create log dir
    logdir = os.path.abspath(hps.logdir) + "/"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    hps.inference = True

    base_dir = hps.restore_path

    checkpoints = { 1.         : 'logs1point0',
                    0.59948425 : 'logs0point599',
                    0.35938137 : 'logs0point359',
                    0.21544347 : 'logs0point215',
                    0.12915497 : 'logs0point129',
                    0.07742637 : 'logs0point077',
                    0.04641589 : 'logs0point046',
                    0.02782559 : 'logs0point027',
                    0.01668101 : 'logs0point016',
                    0.01       : 'logs0point01' }

    model_scopes = { 'church_outdoor' : 'church_model',
                     'bedroom' : 'bedroom_model' }

    # For graphing purposes. Not being used now 
    x_logprobs = []
    y_logprobs = []
    recons = []

    if hps.problem == 'lsun_realnvp':
        _, test_iterator1,_ = get_data(hps, sess, category=hps.category1)
        _, test_iterator2,_ = get_data(hps, sess, category=hps.category2)
    else:
        _, test_iterator, _ = get_data(hps, sess)
        test_iterator1 = test_iterator2 = test_iterator

    mixed, x0, x1, gt0, gt1 = generate_mixture(hps, test_iterator1, test_iterator2, sess)

    for sigma in [1., 0.59948425, 0.35938137, 0.21544347, 0.12915497, 0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01]: 
        # Restore the noisy model(s)
        if hps.problem == 'lsun_realnvp':
            train_iterator1, test_iterator1, data_init1 = get_data(hps, sess, category=hps.category1)
            hps.restore_path = 'finetuned/{}/{}/model_best_loss.ckpt'.format(hps.category1, checkpoints[sigma])
            print("Using checkpoint {}".format(hps.restore_path))
            with tf.variable_scope(model_scopes[hps.category1]):
                model1 = model(sess, hps, train_iterator1, test_iterator1, data_init1, train=False, scope=model_scopes[hps.category1]+'/')

            train_iterator2, test_iterator2, data_init2 = get_data(hps, sess, category=hps.category2)
            hps.restore_path = 'finetuned/{}/{}/model_best_loss.ckpt'.format(hps.category2, checkpoints[sigma])
            print("Using checkpoint {}".format(hps.restore_path))
            with tf.variable_scope(model_scopes[hps.category2]):
                model2 = model(sess, hps, train_iterator2, test_iterator2, data_init2, train=False, scope=model_scopes[hps.category2]+'/')
        else:
            train_iterator, test_iterator, data_init = get_data(hps, sess)
            test_iterator1 = test_iterator2 = test_iterator
            hps.restore_path = 'finetuned/{}/{}/model_best_loss.ckpt'.format(hps.problem, checkpoints[sigma])
            print("Using checkpoint {}".format(hps.restore_path))
            model1 = model2 = model(sess, hps, train_iterator, test_iterator, data_init, train=False)

        # For debugging (comment this stuff out to speed things up)
        #hps.noise_level = sigma
        #estimate_nll(sess, curr_model, hps, test_iterator)
        #hps.noise_level = 0

        x0, x1 = infer(sess, model1, model2, hps, mixed, x0, x1, sigma, x_logprobs, y_logprobs, recons)

        tf.reset_default_graph()
        sess = tensorflow_session()

        # sort by PSNR
        if hps.problem != 'lsun_realnvp':
            psnr, output_to_write = permutations_psnr([x0, x1], [gt0, gt1])
            x0 = output_to_write[0] 
            x1 = output_to_write[1] 

        write_images(hps, x0, 'x_{}.png'.format(sigma))
        write_images(hps, x1, 'y_{}.png'.format(sigma))
        write_images(hps, x0*.5+x1*.5, 'mixed_approx_{}.png'.format(sigma))


def permutations_psnr(output, gt):
    """
    Calculates the psnr of output to gt while handling the per image alignment
        output_x is an array of separated numpy arrays (handles arbitrary many separation)
        gt is an array of the ground truth numpy arrays
    Returns an array of elementwise psnr averages, and an array of reordered images to write
    """
    assert(len(output) == len(gt))
    N = len(output)  # Number of images to separate
    x_to_write = []
    for i in range(N):
        x_to_write.append(output[i].copy())

    all_psnr = []
    for idx in range(output[0].shape[0]):
        best_psnr = -10000
        best_permutation = None

        # Try all permutations of the outputs with the ground truths
        for permutation in permutations(range(N)):
            curr_psnr = sum([psnr(output[permutation[i]][idx], gt[i][idx]) for i in range(N)])
            if curr_psnr > best_psnr:
                best_psnr = curr_psnr
                best_permutation = permutation

        all_psnr.append(best_psnr / float(N))
        for i in range(N):
            x_to_write[i][idx] = output[best_permutation[i]][idx]

    return all_psnr, x_to_write

def psnr(est, gt, trim=True):
    """
    Returns the P signal to noise ratio between the estimate and gt
    Trim means we have to convert both to grayscale and trim to 28 x 28 (MNIST)
    """
    if trim:
        est = est[2:-2, 2:-2,:].mean(2)
        gt = gt[2:-2, 2:-2, :].mean(2)
    return float(-10 * np.log10(((est - gt) ** 2).mean()))

# Get number of training and validation iterations
def get_its(hps):
    # These run for a fixed amount of time. As anchored batch is smaller, we've actually seen fewer examples
    train_its = int(np.ceil(hps.n_train / (hps.n_batch_train * hvd.size())))
    test_its = int(np.ceil(hps.n_test / (hps.n_batch_train * hvd.size())))
    train_epoch = train_its * hps.n_batch_train * hvd.size()

    # Do a full validation run
    if hvd.rank() == 0:
        print(hps.n_test, hps.local_batch_test, hvd.size())
    assert hps.n_test % (hps.local_batch_test * hvd.size()) == 0
    full_test_its = hps.n_test // (hps.local_batch_test * hvd.size())

    if hvd.rank() == 0:
        print("Train epoch size: " + str(train_epoch))
    return train_its, test_its, full_test_its


'''
Create tensorflow session with horovod
'''
def tensorflow_session():
    # Init session and params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Pin GPU to local rank (one GPU per process)
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.Session(config=config)
    return sess

if __name__ == "__main__":
    sys.exit(main())

