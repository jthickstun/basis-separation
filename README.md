# basis-separation
Implementation of the BASIS separation algorithm for source separation with deep generative priors.

by __Vivek Jayaram__ and __John Thickstun__. University of Washington

-------------------------------------------------------------------------------------

This repository provides an implementation of the BASIS (Bayesian Annealed SIgnal Source) separation algorithm.
BASIS separation uses annealed Langevin dynamics to sample from the posterior distribution of source components given a mixed signal.
The codebase is heavily derived from the [NCSN](https://github.com/ermongroup/ncsn)
and [Glow](https://github.com/openai/glow) projects, which provide the models used as priors for BASIS.


## Dependencies

Dependencies are inherited from the NCSN and Glow projects. For NCSN:

* PyTorch

* PyYAML

* tqdm

* pillow

* tensorboardX

* seaborn

For Glow:

* Tensorflow (tested with v1.13.0)

* Horovod (tested with v0.13.8) and (Open)MPI

See the NCSN and Glow repositories for further details.


## Aquiring Pre-Trained Models

The BASIS algorithm requires a trained generative model to use as a prior over source components.
The experiments in this repository explore both NCSN and Glow as priors. Below, we describe where
to find pre-trained NCSN and Glow models and how to set them up for use with BASIS.


### Acquiring NCSN pre-trained models

Pre-trained models for MNIST and CIFAR-10 are provided by the NCSN authors [here](https://drive.google.com/file/d/1BF2mwFv5IRCGaQbEWTbLlAOWEkNzMe5O/view?usp=sharing).
Please extract the downloaded run.zip to the root of your local copy of this repository.
Note that NCSN models are not available for LSUN, so LSUN experiments can only be run with a Glow prior.


### Acquiring Glow pre-trained models

Pre-trained CIFAR-10 and LSUN models are provided by the Glow authors:

* [CIFAR-10](wget https://storage.googleapis.com/glow-demo/logs/abl-1x1-aff.tar)
* [LSUN Bedrooms](wget https://storage.googleapis.com/glow-demo/logs/lsun-rnvp-bdr.tar)
* [LSUN Churches](wget https://storage.googleapis.com/glow-demo/logs/lsun-rnvp-crh.tar)

Unlike NCSN, the Glow models must be fine-tuned with various amounts of Gaussian noise to be used by BASIS.
We suggest the following 10 noise levels:
```
[1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01]
```

You can fine-tune a Glow model using the `noise_level` flag.

For the remainder of this section we will use CIFAR-10 as a running example. We suggest creating a folder in the root repository directory to store the noisy models: `cifar_finetune/`. For consistency, rename the previously-downloaded CIFAR-10 pre-trained model directory to ``logs0point0/`` and store it in ``cifar_finetune/``. To fine-tune the CIFAR-10 model with noise level 0.01, navigate to the `glow` subfolder and run the following:

```
python3 train.py --restore_path ../cifar_finetune/logs0point0/model_best_loss.ckpt --logdir ../cifar_finetune/logs0point01 --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 2 --flow_coupling 1 --seed 20 --learntop --lr 0.001 --noise_level 0.01
```

See the [Glow](https://github.com/openai/glow) repository for further discussion of the flags for `train.py` and instructions for multi-gpu training.

Note that high-noise data may not be well-supported by the zero-noise pre-trained model. If you try to train the higher-noise models by fine-tuning the zero-noise model, you may run into invertibility issues. To resolve this, you can train the models serially: first fine-tune the .01 noise model using the zero-noise model, then fine-tune the 0.01668101 model using the .01 noise model, etc.


## Running BASIS

```
python3 separate.py --model ncsn --runner MnistRunner --config anneal.yml --doc mnist --test --image_folder output
```

```
python3 separate.py --model ncsn --runner CifarRunner --config anneal.yml --doc cifar10 --test --image_folder output
```

```
python3 separate.py --model glow --problem mnist --image_folder output
```

```
python3 separate.py --model glow --problem cifar10 --image_folder output
```

```
python3 separate.py --model glow --problem lsun_realnvp --logdir lsun_output
```


## References

Please see the [NCSN](https://github.com/ermongroup/ncsn) and [Glow](https://github.com/openai/glow) repositories
for the code from which the models in this repository are derived.

