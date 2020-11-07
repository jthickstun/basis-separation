# basis-separation
Implementation of the BASIS separation algorithm described in [Source Separation with Deep Generative Priors](https://arxiv.org/abs/2002.07942).

by [__Vivek Jayaram__](https://www.vivekjayaram.com) and [__John Thickstun__](https://homes.cs.washington.edu/~thickstn/). University of Washington

-------------------------------------------------------------------------------------

This repository provides an implementation of the BASIS (Bayesian Annealed SIgnal Source) separation algorithm.
BASIS separation uses annealed Langevin dynamics to sample from the posterior distribution of source components given a mixed signal.
The codebase is heavily derived from the [NCSN](https://github.com/ermongroup/ncsn)
and [Glow](https://github.com/openai/glow) projects, which provide the models used as priors for BASIS.


## Dependencies

Dependencies are inherited from the NCSN and Glow projects. For NCSN:

* PyTorch (tested with v1.2.0)

* PyYAML (tested with v5.3)

* tqdm (tested with v4.32.1)

* pillow (tested with v6.2.1)

* tensorboardX (tested with v1.6)

* seaborn (tested with v0.9.0)

For Glow:

* Tensorflow (tested with v1.8.0)

* Horovod (tested with v0.13.8) and (Open)MPI

* Keras (tested with v2.2.0)

See the NCSN and Glow repositories for further details on installation of dependencies.


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

* [CIFAR-10](https://storage.googleapis.com/glow-demo/logs/abl-1x1-aff.tar)
* [LSUN Bedrooms](https://storage.googleapis.com/glow-demo/logs/lsun-rnvp-bdr.tar)
* [LSUN Churches](https://storage.googleapis.com/glow-demo/logs/lsun-rnvp-crh.tar)

Unlike NCSN, the Glow models must be fine-tuned with various amounts of Gaussian noise to be used by BASIS. Fine-tuned models for MNIST, CIFAR-10, and LSUN Churches and Bedrooms can be downloaded here:

* [MNIST](https://drive.google.com/uc?export=download&id=1WfyOlE_G5goFYo0WH_P7Tq4tS2S3y-EC) (~5Gb)
* [CIFAR-10](https://drive.google.com/uc?export=download&id=1BGJHqso8C8GGI7H8KmMBhio07gQFdQLI) (~7Gb)
* [LSUN Bedrooms](https://drive.google.com/uc?export=download&id=1vmVMP7QOdyvD95-6WrczVi_kNK2silg3) (~13Gb)
* [LSUN Churches](https://drive.google.com/uc?export=download&id=1yNUk2jnTFNPQiOxmlrgxjn9gviKnncQe) (~13Gb)

Please extract the downloaded tarballs to the root of your local copy of this repository.
The finetuned models will be extracted to the finetuned/ sub-directory, where they can be found by the BASIS separation script.

### Fine-tuning a Glow model

You can fine-tune a Glow model using the `noise_level` flag.

For the remainder of this section we will use CIFAR-10 as a running example. We suggest creating a folder in the root repository directory to store the noisy models: `cifar_finetune/`. For consistency, rename the previously-downloaded CIFAR-10 pre-trained model directory to ``logs0point0/`` and store it in ``cifar_finetune/``. To fine-tune the CIFAR-10 model with noise level 0.01, navigate to the `glow` subfolder and run the following:

```
python3 train.py --restore_path ../cifar_finetune/logs0point0/model_best_loss.ckpt --logdir ../cifar_finetune/logs0point01 --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 2 --flow_coupling 1 --seed 20 --learntop --lr 0.001 --noise_level 0.01
```

See the [Glow](https://github.com/openai/glow) repository for further discussion of the flags for `train.py` and instructions for multi-gpu training.

Note that high-noise data may not be well-supported by the zero-noise pre-trained model. If you try to train the higher-noise models by fine-tuning the zero-noise model, you may run into invertibility issues. To resolve this, you can train the models serially: first fine-tune the .01 noise model using the zero-noise model, then fine-tune the 0.01668101 model using the .01 noise model, etc.


## Running BASIS

See the previous section for instructions about downloaded pre-trained models for running BASIS. Below are some examples of how to run BASIS separation with each of the models and datasets discussed in the paper.

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

## Colorization Example
NCSN only for now
```
python3 separate.py --model ncsn --runner CifarRunner --config anneal.yml --doc cifar10 --test --image_folder output
```

### CIFAR-10 Example (using NCSN)
![](assets/cifar_video.gif)

### LSUN Example (using Glow)
![](assets/lsun_video.gif)

## References

Please see the [NCSN](https://github.com/ermongroup/ncsn) and [Glow](https://github.com/openai/glow) repositories
for the code from which the models in this repository are derived.

To reference this work, please cite

```bib
@article{jayaram2020source,
  author    = {Vivek Jayaram and John Thickstun},
  title     = {Source Separation with Deep Generative Priors},
  journal   = {International Conference on Machine Learning},
  year      = {2020},
}
```
