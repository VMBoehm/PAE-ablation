# PAE-ablation
Ablation studies between the Probabilistic AutoEncoder (PAE), flow-VAE and &beta;<sub>0</sub>-VAE.

This respository is an updated version of [the PAE repository](https://github.com/VMBoehm/PAE) with additional ablation studies between the PAE and other VAE models. It contains updated code compatible with tensorflow 2 and features more options for training (V)AE models. Additions include, among others, a VAE with a normalizing flow prior. For a detailed explanation of the PAE model, please visit https://github.com/VMBoehm/PAE.

## Why this new repository?
We wanted to better understand the performance of the PAE model by means of a series of ablation studies. Formerly we had only compared the PAE to &beta;-VAEs.
In this repository we provide code from our ablation studies against 
- **flow-VAE**: A VAE with a flow prior. This allows us to compare it to a PAE of identical architecture.
- **&beta;<sub>0</sub>-VAE**: A VAE model with a flow prior, that similarly to the PAE is trained in two stages.

## Installation
The PAE package can be installed the root directory with
``` pip install -e .```
All requirements are listed in *tf22.yml*

## Model Training
We provide a python package that automizes the AE (the first stage of the PAE) and VAE trainings.   
Running   

```python main_tf2.py --helpfull```   

displays all the options for running the code. 
Parameters include the data set, the latent space dimensionality, locations for storing results, the type of loss (AE or VAE), training parameters etc.
To train a VAE with a normalizing flow prior the simplest command is

```python main_tf2.py --loss=VAE --flow_prior=True```, 

to train the equivalent autoencoder the command is

```python main_tf2.py --loss=AE --dropout_rate=0.15```.

The models train by default on Fashion-MNIST and for 300 000 training steps, the default parameters are set to reproduce the results from our paper. 

We provide additional notebooks for training the PAE second stage, the normalizing flow. These can be found [here](https://github.com/VMBoehm/PAE-ablation/tree/master/notebooks/FlowTraining).

If you are looking for a well commented, Google Colab compatible template to train a normalizing flow with your choice of RealNVP, Neural Spline Flow and Glow layers, have a look at https://github.com/VMBoehm/PAE/blob/master/TrainNVP_simplified_and_explained.ipynb.

## Pretrained Models
Pretrained models can be downloaded [here](https://github.com/VMBoehm/PAE-ablation/tree/master/notebooks/FlowTraining)

## Model Evaluation
We provide notebboks and python scripts for model evaluation.

### Reconstruction Error
To measure the reconstruction error and related statistics for each model, run notebooks in [this] folder(https://github.com/VMBoehm/PAE-ablation/tree/master/notebooks/Reconstructions)

### FID Scores
Python scripts for measuring sample quality in terms of FID scores for each model can be found [here](https://github.com/VMBoehm/PAE-ablation/tree/master/scripts/FIDScores)

### Out-of-Distribution Error


### Image Imputation

### Generalization
