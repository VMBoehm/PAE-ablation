#!/usr/bin/env python
# coding: utf-8

# # Out-of-Distributin detection on different OoD and with different metrics

# In[1]:


import tensorflow.compat.v1 as tf
# #To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
#tf.disable_eager_execution()
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
import sys
import pickle
from functools import partial


plt.rcParams.update({'font.family' : 'lmodern', 'font.size': 16,                                                                                                                                                    
                     'axes.labelsize': 16, 'legend.fontsize': 12, 
                     'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.titlesize': 16,
                     'axes.linewidth': 1.5}) 


# In[2]:


import scipy


# In[3]:


import tensorflow_probability as tfp
import tensorflow_hub as hub
tfd = tfp.distributions
tfb = tfp.bijectors


# In[4]:


from pae.model_tf2 import get_prior


# In[5]:


import pae.create_datasets as crd
import pae.load_data as ld
load_funcs=dict(mnist=ld.load_mnist, fmnist=ld.load_fmnist)


# In[6]:


import os
import pickle
PROJECT_PATH = '/global/u2/v/vboehm/codes/PAE/'
PARAMS_PATH = os.path.join(PROJECT_PATH,'params')

param_file = 'params_fmnist_-1_40_infoGAN_VAE_best_params_noaugment_full_sigmaVAE_beta0'
params      = pickle.load(open(os.path.join(PARAMS_PATH,param_file+'.pkl'),'rb'))


# In[7]:


params['data_dir']

ood_set='mnist'
flip=''

# In[8]:


load_func                                          = partial(load_funcs[params['data_set']])
x_train, y_train, x_valid, y_valid, x_test, y_test = load_func(params['data_dir'],flatten=False)

if np.all(x_test)==None:
    x_test=x_valid

x_train    = x_train/255.-0.5
x_test     = x_test/255.-0.5
x_valid    = x_valid/255.-0.5


# In[9]:


def get_outliers(dataset,flip='no'):
    if dataset=='omniglot':
        import tensorflow_datasets as tfds
        from skimage.transform import resize
        omni= tfds.load('omniglot')
        glot = tfds.as_numpy(omni)
        samples=[]
        for sample in glot['test']:
            s = resize(sample['image'],(28,28))
            samples.append(-(s-0.5))
        samples=np.asarray(samples)
        samples = np.mean(samples[0:10000],axis=-1)
        x_valid_ood=np.expand_dims(samples,-1)
        x_test_ood=x_valid_ood
    else:
        load_func                                         = partial(load_funcs[dataset])
        x_train_ood, y_train, x_valid_ood, y_valid, x_test_ood, y_test = load_func(params['data_dir'],flatten=False)

        if np.any(x_test_ood)is None:
            x_test_ood=x_valid_ood

        if  flip=='horizontal':
            x_test_ood    = np.asarray([np.fliplr(x) for x in x_test_ood/255.-0.5])
        elif flip=='vertical':
            x_test_ood    = np.asarray([np.flipud(x) for x in x_test_ood/255.-0.5])
        else:
            x_test_ood    = x_test_ood/255.-0.5

    for ii in range(2):
        plt.imshow(np.squeeze(x_test_ood[ii]),cmap='gray')
        plt.axis('off')
        plt.show()
    return x_test_ood


# In[10]:


plt.imshow(np.squeeze(x_test[1]),cmap='gray')
plt.axis('off')
plt.show()


# ### To reproduce results choose either 'mnist', 'fmnist', or 'omniglot'. To flip use keyword 'horizontal' or 'vertical'

# In[27]:
from sklearn.utils import resample


x_valid_ood=get_outliers(ood_set, flip=flip)


# In[12]:


generator_path   = os.path.join(params['module_dir'],'decoder')
encoder_path     = os.path.join(params['module_dir'],'encoder')
nvp_path         = os.path.join(params['module_dir'],'tag7_130')


# In[13]:


def get_likelihood(decoder,sigma):
  
    def likelihood(z):
        mean = decoder({'z':z})['x']
        return tfd.Independent(tfd.MultivariateNormalDiag(loc=mean,scale_diag=sigma))

    return likelihood

def get_posterior(encoder):

    def posterior(x):
        mu, sigma        = tf.split(encoder({'x':x})['z'], 2, axis=-1)
        sigma            = tf.nn.softplus(sigma) + 1e-6
        approx_posterior = tfd.MultivariateNormalLinearOperator(loc=mu,scale=tf.linalg.LinearOperatorDiag(sigma))

        #approx_posterior = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return approx_posterior

    return posterior


# In[14]:


sigma         = params['full_sigma']
sigma         = tf.cast(sigma,tf.float32)
print(sigma.shape)
encoder       = hub.KerasLayer(encoder_path,trainable=False, signature_outputs_as_dict=True)
decoder       = hub.KerasLayer(generator_path, trainable=False, signature_outputs_as_dict=True)
nvp_funcs     = hub.KerasLayer(nvp_path, trainable=False, signature_outputs_as_dict=True)


# In[15]:


likelihood       = get_likelihood(decoder,sigma)
prior            = get_prior(params['latent_size'])
posterior        = get_posterior(encoder)


# In[16]:


def bwd_pass(z):
    return nvp_funcs({'z_sample':z,'sample_size':1, 'u_sample':np.zeros((1,params['latent_size']))})['bwd_pass']

def fwd_pass(u):
    return nvp_funcs({'z_sample':np.zeros((1,params['latent_size'])),'sample_size':1, 'u_sample':u})['fwd_pass']


def compute_nvp_prior(z):
    return nvp_funcs({'z_sample':z,'sample_size':1, 'u_sample':np.zeros((1,params['latent_size']))})['log_prob']

def inverse_log_det_jacobian(z):
    MAP_Gauss        = prior.log_prob(bwd_pass(z))
    MAP_prior        = nvp_funcs({'z_sample':z,'sample_size':1, 'u_sample':np.zeros((1,params['latent_size']))})['log_prob']
    # the Jacobian should be the difference
    NF_Jac           = MAP_prior - MAP_Gauss
    return NF_Jac

def get_encoded(x):
    mu, sig = tf.split(encoder({'x':x})['z'],2, axis=-1)
    return mu

def get_sigma(x):
    mu, sig = tf.split(encoder({'x':x})['z'],2, axis=-1)
    return sig

def get_decoded(z):
    return decoder({'z':z})['x']

def prior_eval(z,nvp_funcs=nvp_funcs):
    prior         = nvp_funcs({'z_sample':z,'sample_size':1, 'u_sample':np.zeros((1,params['latent_size']))})['log_prob']
    return prior


# In[17]:


def get_pz(data):
    mu = get_encoded(data)
    return prior_eval(mu)
    


# In[18]:


def get_pxz(data):
    MAP = get_encoded(data)
    return likelihood(MAP).log_prob(data)

def get_recon_error(data):
    z = get_encoded(data)
    decoded = get_decoded(z)
    recon = -tf.reduce_mean(tf.square(decoded-data),axis=(1,2))
    return recon

def get_joint(data):
    return get_pz(data)+get_pxz(data)

def get_entropy(data):
    return -posterior(data).entropy()


# In[19]:


def get_second_KL_term(data, sample_size=10):
    sample = posterior(data).sample(sample_size)
    p_z    = prior_eval(sample[0],nvp_funcs=nvp_funcs)
    for ii in range(sample_size-1):
        p_z+=prior_eval(sample[ii+1],nvp_funcs=nvp_funcs)
    p_z/=sample_size
    return p_z


# In[20]:


def get_likelihood_term(data, sample_size=10):
    sample = posterior(data).sample(sample_size)
    sample_likelihood = likelihood(sample[0]).log_prob(data)
    for ii in range(sample_size-1):
        sample_likelihood+=likelihood(sample[ii+1]).log_prob(data)
    sample_likelihood/=sample_size
    return sample_likelihood


# In[21]:


def get_KL(data, sample_size=10):
    term1 = get_entropy(data)
    term2 = get_second_KL_term(data, sample_size=sample_size)
    return -(term1-term2)


# In[22]:


def get_ELBO(data, sample_size=10):
    likelihood = get_likelihood_term(data, sample_size)
    
    KL = get_KL(data, sample_size)
    return likelihood+KL


from scipy.integrate import simps

metrics = ['ELBO']#, 'likelihood','KL', 'entropy', 'cross entropy']#'prior','likelihood', 'recon_error', 'Laplace_without_Volume']
labels  = ['ELBO']#, 'likelihood','KL', 'entropy', 'cross entropy']#r'$\mathrm{log}\, p(z)$',r'$\mathrm{log}\, p(x|z)$', 'reconstruction error',r'$\mathrm{log}\, p(z)+\mathrm{log}\, p(x|z)$' ]
objs    = [get_ELBO, get_likelihood_term, get_KL, lambda x: -get_entropy(x), get_second_KL_term]#get_pz, get_pxz, get_recon_error, get_joint]


sig = np.maximum(params['full_sigma'],10)
for jj in range(len(metrics)):
    AUROC=[]
    for nn in range(100):
        x_val_sample = resample(x_valid,replace=True,n_samples=10000)
        x_val_sample_ood = resample(x_valid_ood,replace=True,n_samples=10000)
        objective=[]

        for ii in range(len(x_val_sample)//params['batch_size']):
            data_sample = x_val_sample[ii*params['batch_size']:(ii+1)*params['batch_size']]
            objective+=[objs[jj](data_sample)]
        delta = len(x_val_sample)%params['batch_size']
        data_sample = np.concatenate((x_val_sample[(ii+1)*params['batch_size']:len(x_val_sample)],x_val_sample[0:params['batch_size']-delta]))
        objective+=[objs[jj](data_sample)]
        objective = np.asarray(objective).flatten()[0:len(x_val_sample)]
        objective_ood=[]
        for ii in range(len(x_val_sample_ood)//params['batch_size']):
            data_sample = x_val_sample_ood[ii*params['batch_size']:(ii+1)*params['batch_size']]
            objective_ood+=[objs[jj](data_sample)]
        delta = len(x_val_sample)%params['batch_size']
        data_sample = np.concatenate((x_val_sample_ood[(ii+1)*params['batch_size']:len(x_val_sample_ood)],x_val_sample_ood[0:params['batch_size']-delta]))
        objective_ood+=[objs[jj](data_sample)]
        objective_ood = np.asarray(objective_ood).flatten()[0:len(x_val_sample)]

        objective = np.asarray(objective)
        objective_ood = np.asarray(objective_ood)
        objs_ = np.sort(objective)
        objs_ood_ = np.sort(objective_ood)
        false_pos=[]
        true_pos=[]
        for ii in range(10000):
            val = objs_[ii]
            true_pos.append(len(np.where(objs_>=val)[0])/len(objs_))
            false_pos.append(len(np.where(objs_ood_>=val)[0])/len(objs_))
        false_pos = np.asarray(false_pos,dtype=np.float64)
        AUROC+=[1-np.sum(false_pos)*1./np.float(len(objs_))]
    print(metrics[jj], np.mean(AUROC), np.std(AUROC))
#false_positive.append(false_pos)

# In[ ]:




