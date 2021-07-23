#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
import sys
import pickle
from functools import partial
 
plt.rcParams.update({'font.size': 16,                                                                                                                                                    
                     'axes.labelsize': 16, 'legend.fontsize': 12, 
                     'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.titlesize': 16,
                     'axes.linewidth': 1.5}) 

print(tf.__version__)


# In[2]:


import pae.load_data as ld
load_funcs=dict(mnist=ld.load_mnist, fmnist=ld.load_fmnist, celeba=ld.load_celeba)


# In[3]:


from pae.fid_score_tf2 import *


# In[4]:


PROJECT_PATH = "../../" 
PARAMS_PATH  = os.path.join(PROJECT_PATH,'params')
param_file   = 'params_fmnist_-1_40_infoGAN_VAE_best_params_noaugment_sigmafixed_full_sigmaVAE_beta0'
params       = pickle.load(open(os.path.join(PARAMS_PATH,param_file+'.pkl'),'rb'))

#flow = 'tag7_130'


# In[5]:


ROOT  = '/global/cscratch1/sd/vboehm/RNF/'
LOCAL = '/global/homes/v/vboehm/codes/PAE'


# In[6]:


params['module_dir'] = params['module_dir'].replace('/global/scratch/vboehm/rnf/',ROOT)
params['data_dir'] = '/global/cscratch1/sd/vboehm/Datasets'
params['plot_dir'] = os.path.join(LOCAL,'plots',params['label'])
pickle.dump(params, open(os.path.join(PARAMS_PATH,param_file+'.pkl'),'wb'))


# In[7]:


print(params['plot_dir'])


# In[8]:


if not os.path.isdir(params['plot_dir']):
    os.makedirs(params['plot_dir'])
# if not os.path.isdir(params['data_dir']):
#     os.makedirs(params['data_dir'])


# In[9]:


load_func                                          = load_funcs[params['data_set']]
x_train, y_train, x_valid, y_valid, x_test, y_test = load_func(params['data_dir'])

if np.all(x_test)==None:
    x_test=x_valid

x_train    = x_train/255.-0.5
x_test     = x_test/255.-0.5
x_valid    = x_valid/255.-0.5


# In[10]:


generator_path   = os.path.join(params['module_dir'],'beta0VAE_decoder')
encoder_path     = os.path.join(params['module_dir'],'beta0VAE_encoder')
nvp_path         = os.path.join(params['module_dir'],'beta0VAE_flow')


# In[11]:


shape = params['data_shape']
shape = np.append(-1,shape)
print(shape)

x_valid = np.reshape(x_valid, shape)


# In[12]:


import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import tensorflow_hub as hub
tfd = tfp.distributions
tfb = tfp.bijectors


# In[13]:


tf.reset_default_graph()

data          = tf.placeholder(shape=[None]+params['data_shape'],dtype=tf.float32)
latent_point  = tf.placeholder(shape=[None,params['latent_size']],dtype=tf.float32)
sample_size   = tf.placeholder_with_default(params['batch_size'], shape=[])
encoder       = hub.Module(encoder_path, trainable=False)
decoder       = hub.Module(generator_path, trainable=False)
nvp           = hub.Module(nvp_path, trainable=False)
prior         = tfd.MultivariateNormalDiag(tf.zeros(params['latent_size']), scale_identity_multiplier=1.0)

encoded, _    = tf.split(encoder({'x':data},as_dict=True)['z'], 2, axis=-1)
decoded       = decoder({'z':encoded},as_dict=True)['x']

decoded_latent= decoder({'z':latent_point},as_dict=True)['x']

sigma         = tf.reduce_mean(tf.sqrt(tf.square(data-decoded)),axis=0)

samples         = prior.sample(sample_size)
#decoded_samples = decoder({'z':samples},as_dict=True)['x']
nvp_samples     = nvp({'z_sample':np.zeros((1,params['latent_size'])),'sample_size':1, 'u_sample':samples}, as_dict=True)['fwd_pass']
u_samples       = nvp({'z_sample':encoded,'sample_size':1, 'u_sample':samples}, as_dict=True)['bwd_pass']
decoded_nvp_samples = decoder({'z':nvp_samples},as_dict=True)['x']


# In[14]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[15]:


recs=[]
delta = len(x_valid)%params['batch_size']
for ii in range(len(x_valid)//params['batch_size']):
    recs+=[sess.run(decoded, feed_dict={data:x_valid[ii*params['batch_size']:(ii+1)*params['batch_size']].reshape((-1,28,28,1))})]
data_sample=np.concatenate((x_valid[(ii+1)*params['batch_size']:len(x_valid)],x_valid[0:params['batch_size']-delta]))
recs+=[sess.run(decoded, feed_dict={data:data_sample.reshape((-1,28,28,1))})]
recs = np.asarray(recs).reshape((-1,784))[0:len(x_valid)]




from sklearn.utils import resample
 




recs = np.reshape(recs,(-1,28,28,1))


random_nvp_samples=[]
for ii in range(50000//params['batch_size']+1):
    random_nvp_samples+=[sess.run(decoded_nvp_samples)]
random_nvp_samples=np.asarray(random_nvp_samples).reshape((-1,28,28))[0:50000]


def evaluate_fid_score(fake_images, real_images,norm=True):
    #np.random.shuffle(real_images)
    assert(len(real_images))
    assert(len(fake_images))
    real_images = real_images[0:10000]
    fake_images = fake_images[0:10000]
    real_images = preprocess_fake_images(real_images, norm)
    fake_images = preprocess_fake_images(fake_images, norm)

    inception_path = check_or_download_inception()

    create_inception_graph(inception_path)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    print('calculating tf features...')
    real_out = get_activations_tf(real_images, sess)
    fake_out = get_activations_tf(fake_images, sess)
    fid_result = fid_score(real_out, fake_out)
    sess.close()

    return fid_result


# In[32]:


import warnings
warnings.filterwarnings('ignore')


# In[33]:


import sklearn
from sklearn.utils import resample




#fid_scores = []
#for ii in range(50):
#    print(ii)
#    sample_1 = resample(recs,replace=True,n_samples=10000)
#    sample_2 = resample(x_valid,replace=True,n_samples=10000)
#    fid_scores.append(evaluate_fid_score(sample_1,sample_2))
    
#fid_scores=np.asarray(fid_scores)
#_ = plt.hist(fid_scores, density=True, bins=100)
#plt.show()
#print('recon:', np.mean(fid_scores),np.std(fid_scores))

fid_scores = []
for ii in range(50):
    print(ii)
    sample_1 = resample(np.expand_dims(random_nvp_samples,-1),replace=True,n_samples=10000)
    sample_2 = resample(x_valid,replace=True,n_samples=10000)
    fid_scores.append(evaluate_fid_score(sample_1,sample_2))
    
fid_scores=np.asarray(fid_scores)
_ = plt.hist(fid_scores, density=True, bins=100)
plt.show()
print('samples:', np.mean(fid_scores),np.std(fid_scores))
