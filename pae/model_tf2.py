"""
Copyright 2019 Vanessa Martina Boehm

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import rnf.networks_tf2 as nw

### these two functions are inspired by https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py and modified to work with flattened data by adding a shape keyword

def pack_images(images, rows, cols,shape):
    """Helper utility to make a field of images.
    Borrowed from Tensorflow Probability
    """
def make_images(images, nrows, ncols,shape,params):
    width  = shape[-3]
    height = shape[-2]
    depth  = shape[-1]
    bsize  = tf.shape(input=images)[0]
    images = tf.reshape(images, (-1, width, height, depth))
    nrows  = tf.minimum(nrows, bsize)
    ncols  = tf.minimum(bsize//nrows, ncols)
    images = images[:nrows * ncols]
    images = tf.reshape(images, (nrows, ncols, width, height, depth))
    images = tf.transpose(a=images, perm=[0, 2, 1, 3, 4])
    if params['offset']:
        images = images+0.5
    images = tf.clip_by_value(tf.reshape(images, [1, nrows * width, ncols * height, depth]), 0, 1)
    return images


def image_tile_summary(name, tensor, rows, cols, shape, params):
    tf.compat.v1.summary.image(name, make_images(tensor, rows, cols, shape, params), max_outputs=1)
#######

def get_prior(latent_size):
    return tfd.MultivariateNormalDiag(tf.zeros(latent_size), scale_identity_multiplier=1.0)

def get_GN_covariance(decoded,z,params):

    ones  = tf.linalg.eye(params['latent_size'], batch_shape=[params['batch_size']],dtype=tf.float32) 
    
    with tf.compat.v1.variable_scope("likelihood", reuse=tf.compat.v1.AUTO_REUSE):
        sigma = tf.compat.v1.get_variable(name='sigma', use_resource=False, initializer=tf.ones([np.prod(params['output_size'])])*params['sigma'])
    decoded = tf.reshape(decoded,[params['batch_size'],-1])
    grad_g  = tf.gather(tf.gradients(ys=decoded/sigma,xs=z),0)
    grad_g2 = tf.einsum('ij,ik->ijk',grad_g,grad_g)
    GNhess  = ones#+grad_g2
    cov     = tf.linalg.inv(GNhess)
    cov     = 0.5*(cov+tf.linalg.matrix_transpose(cov))
    det     = tf.linalg.det(cov)

#    hess    = tf.hessians(decoded,z)
#    hess    = tf.gather(hess, 0)
#    hess    = tf.reduce_sum(hess, axis = 2 )
#    hess    = 0.5*(hess+tf.linalg.transpose(hess))
#    detC    = 1./(tf.linalg.det(hess))

    return cov, det#, detC



def get_posterior(encoder,params):

    def posterior(x):
        mu, sigma        = tf.split(encoder({'x':x},as_dict=True)['z'], 2, axis=-1)
        if params['constant_post_var']:
            sigma            = tf.ones(tf.shape(sigma), dtype=tf.float32)
        else:
            sigma            = tf.nn.softplus(sigma) + 1e-6
        approx_posterior = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return approx_posterior

    return posterior


def get_Bernoulli_likelihood(decoder,params):

    def likelihood(z):
        mean = decoder({'z':z},as_dict=True)['x']
        mean = tf.reshape(mean,[params['batch_size'],-1])

        return tfd.Independent(tfd.Bernoulli(probs=mean))

    return likelihood

def get_likelihood(decoder,sigma,params):

    #with tf.variable_scope("likelihood", reuse=tf.AUTO_REUSE):
    #    sigma = tf.get_variable(name='sigma', initializer=tf.ones([np.prod(params['output_size'])])*params['sigma'])

    def likelihood(z):
        mean = decoder({'z':z},as_dict=True)['x']
        mean = tf.reshape(mean,[params['batch_size'],-1])
        return tfd.Independent(tfd.MultivariateNormalDiag(loc=mean,scale_diag=sigma*tf.ones([np.prod(params['output_size'])])))
        
    return likelihood






def get_laplace_posterior(z,decoder,params):

    decoded          = decoder({'z':z},as_dict=True)['x']
    cov, det         = get_GN_covariance(decoded,z,params)
    laplace_posterior= tfd.MultivariateNormalFullCovariance(loc=z,covariance_matrix=cov)

    return laplace_posterior, det

def model_fn(features, labels, mode, params, config):
    del labels, config
    try:
        features = features['x']
    except:
        pass

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    encoder          = nw.make_encoder(params, is_training)
    decoder          = nw.make_decoder(params, is_training)

    if params['flow_prior']:
        flow_prior = nw.make_nvp(params)
    
    global_step      = tf.compat.v1.train.get_or_create_global_step()

    prior            = get_prior(params['latent_size'])

    posterior        = get_posterior(encoder,params)
    approx_posterior = posterior(features)
    MAP              = approx_posterior.mean()

    decoded          = decoder({'z':MAP},as_dict=True)['x']

    chi2             = tf.stop_gradient(tf.reshape(tf.square(decoded-features),[params['batch_size'],-1]))
    
    #first taking sqrt and then averaging over batch (correct?)    	
    #sigma_pixel      = tf.reduce_mean(tf.sqrt(chi2),axis=0)
    #sigma_mean       = tf.reduce_mean(tf.sqrt(tf.reduce_mean(chi2,axis=1)),axis=0)
    #sigma_mean       = sigma_mean*tf.ones([np.prod(params['output_size'])])

    #first averaging over batch then sqrt      
    sigma_pixel      = tf.sqrt(tf.reduce_mean(chi2,axis=0))
 
    
    sigma_mean       = tf.sqrt(tf.reduce_mean(chi2))

    
    if params['sigma_annealing']:
        sigma = tf.cond(global_step<200000,lambda: 1.-tf.cast(global_step,tf.float32)/200000.*(1.-params['sigma']), lambda: params['sigma'])
    else:
        sigma = params['sigma']

    if params['likelihood']=='Gauss':
        likelihood       = get_likelihood(decoder,sigma,params)
    else:
        likelihood       = get_Bernoulli_likelihood(decoder,params)
    
    map_likelihood    = likelihood(MAP).log_prob(tf.reshape(features,[params['batch_size'],-1]))

    posterior_sample  = approx_posterior.sample()

    if params['flow_prior']:
        kl1 = -approx_posterior.entropy() 
        kl2 = flow_prior({'z_sample':posterior_sample,'sample_size':1, 'u_sample':np.zeros((1,params['latent_size']))}, as_dict=True)['log_prob']
        kl  = kl1-kl2 
    else:
        kl               = 0.5 * tf.reduce_sum(tf.square(approx_posterior.scale.diag) + tf.square(approx_posterior.loc) - tf.math.log(tf.square(approx_posterior.scale.diag))-1, axis=-1)
    
    kl_lambda         = tf.maximum(kl - params['lambda'], 0.0)
    sample_likelihood = likelihood(posterior_sample).log_prob(tf.reshape(features,[params['batch_size'],-1]))
    
    objective_AE     = map_likelihood

    if params['C_annealing']:
        C = tf.cond(global_step<100000,lambda: tf.cast(global_step,tf.float32)/100000.*params['C'], lambda: params['C'])
    else:
        C = params['C']

    if params['beta_annealing']:
        beta = tf.cond(global_step<100000,lambda: (1.-tf.cast(global_step,tf.float32)/100000.)*(params['beta']-1.)+1., lambda:1.)
    else:
        beta = params['beta']

    if params['free_bits']:
        objective_VAE    = sample_likelihood-beta*kl_lambda
    else:
        objective_VAE    = sample_likelihood-beta*tf.math.abs(kl-C)

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        tf.compat.v1.summary.scalar('log_likelihood',tf.reduce_mean(map_likelihood))
        if params['loss']=='VAE':
            tf.compat.v1.summary.scalar('beta',beta)
            tf.compat.v1.summary.scalar('KL_divergence',tf.reduce_mean(kl))
            tf.compat.v1.summary.scalar('KL_divergence_beta',tf.reduce_mean(kl*beta))
            tf.compat.v1.summary.scalar('free_bits_kl', tf.reduce_mean(kl_lambda))
            tf.compat.v1.summary.scalar('elbo',tf.reduce_mean(objective_VAE))
            tf.compat.v1.summary.scalar('sigma_train',sigma)
        tf.summary.scalar('sigma', tf.reduce_mean(sigma_mean))

        if params['loss']=='VAE':
            loss         = -tf.reduce_mean(objective_VAE)
        else:
            loss         = -tf.reduce_mean(objective_AE)

        #all_vars         = tf.trainable_variables()
        #train_vars       = [var for var in all_vars if 'sigma' not in var.name]

        lr               = tf.cond(global_step<(4*params['max_steps']//5), lambda: params['learning_rate'], lambda: params['learning_rate']/10.)

        tf.compat.v1.summary.scalar('learning_rate',lr)

        optimizer        = tf.compat.v1.train.AdamOptimizer(lr)

        train_op         = optimizer.minimize(loss,global_step=global_step)

        if params['output_images']:
            if mode == tf.estimator.ModeKeys.TRAIN:
                image_tile_summary('training/inputs',features, rows=4, cols=4, shape=params['data_shape'], params=params)
                image_tile_summary('training/reconstructions',decoded, rows=4, cols=4, shape=params['data_shape'],params=params)
            else:
                image_tile_summary('test/inputs',features, rows=4, cols=4, shape=params['data_shape'], params=params)
                image_tile_summary('test/reconstructions',decoded, rows=4, cols=4, shape=params['data_shape'], params=params)
            if params['flow_prior']:
                samples  = flow_prior({'z_sample':np.zeros((1,params['latent_size'])),'sample_size':params['batch_size'], 'u_sample':np.zeros((1,params['latent_size']))}, as_dict=True)['sample']
                samples = tf.reshape(samples,[params['batch_size'],params['latent_size']])
                samples = decoder({'z':samples},as_dict=True)['x']

            else:
                samples  = decoder({'z':prior.sample(params['batch_size'])},as_dict=True)['x']
            image_tile_summary('prior_samples',samples, rows=4, cols=4, shape=params['data_shape'], params=params)  

        eval_metric_ops={
                'log_likelihood': tf.compat.v1.metrics.mean(map_likelihood),
        #        'log_prior_at_MAP': tf.metrics.mean(map_prior),
        #        'KL_divergence': tf.metrics.mean(kl),
                'sigma':tf.compat.v1.metrics.mean(sigma_mean),
        #        'elbo':tf.metrics.mean(objective_VAE),
        #        'KL_laplace_prior':tf.metrics.mean(kl_laplace_prior),
        #        'KL_laplace_vmf':tf.metrics.mean(kl_laplace_vmf),
        #        'Gauss_Newton_determinant': tf.metrics.mean(det),
                #'full_Hessian_determinant': tf.metrics.mean(detC),
        }

        eval_summary_hook = tf.estimator.SummarySaverHook(save_steps=1,output_dir=params['model_dir'],summary_op=tf.compat.v1.summary.merge_all())
        evaluation_hooks  = [eval_summary_hook]
        
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops = eval_metric_ops, evaluation_hooks=evaluation_hooks)
    else:
        predictions = {'likelihood_pred':tf.reduce_mean(map_likelihood)}
    
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
