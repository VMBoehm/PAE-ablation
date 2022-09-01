#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import seaborn as sns
import seaborn_image
import torch.nn as nn
from sinf import GIS
import os
import scipy
import sklearn
import sklearn.metrics as metrics
import pickle
import pandas

from models_cifar import ToFloatTensor2D, Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

code_length = 64
random_seed = 123
batch_size  = 64
learning_rate = 0.001
EPOCHS      = 200

print(code_length,EPOCHS)

for index in np.arange(0,1):
    print(index)
    cifar_train = torchvision.datasets.CIFAR10('/global/cscratch1/sd/vboehm/Datasets', train=True, transform = ToFloatTensor2D(device), download=True)
    cifar_test  = torchvision.datasets.CIFAR10('/global/cscratch1/sd/vboehm/Datasets', train=False, transform = ToFloatTensor2D(device), download=True)
    cifar_valid = torchvision.datasets.CIFAR10('/global/cscratch1/sd/vboehm/Datasets', train=False, transform = ToFloatTensor2D(device), download=True)

    idx = np.asarray(cifar_valid.targets) == index
    cifar_valid.targets = np.asarray(cifar_valid.targets)[idx]
    cifar_valid.data    = cifar_valid.data[idx]

    idx1 = np.asarray(cifar_test.targets) == index
    idx2 = np.asarray(cifar_test.targets) != index
    cifar_test.targets = np.asarray(cifar_test.targets)
    cifar_test.targets[idx1] = torch.ones(np.asarray(cifar_test.targets)[idx1].shape,dtype=torch.int64)
    cifar_test.targets[idx2] = torch.zeros(np.asarray(cifar_test.targets)[idx2].shape,dtype=torch.int64)




    train_dataloader = DataLoader(cifar_train, batch_size=64, shuffle=True)
    test_dataloader  = DataLoader(cifar_test, batch_size=64, shuffle=True)
    valid_dataloader  = DataLoader(cifar_valid, batch_size=64, shuffle=True)



    model = Model(code_length).to(device)

    loss_fn   = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    best_vloss = 1_000_000.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.00001)
    
    
    def train_one_epoch(epoch_index):
        running_loss = 0
        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(train_dataloader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(inputs, outputs)
            loss.backward()
            running_loss+=loss.item()

            # Adjust learning weights
            optimizer.step()
        running_loss/=(i+1)   

        return running_loss

    losses = []
    epoch_number = 0
    while epoch_number<EPOCHS:
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0
        for i, vdata in enumerate(valid_dataloader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(vinputs,voutputs)
            running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)
        scheduler.step(avg_vloss)
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = '/global/cscratch1/sd/vboehm/AbatiExperiments/cifar_model_{}_{}_{}'.format(timestamp, index,code_length)
            torch.save(model.state_dict(), model_path)
            no_improv = 0
        else:
            no_improv+=1
            
        print(avg_loss, avg_vloss)
        losses.append([avg_loss, avg_vloss])
        if no_improv==25:
            break
        epoch_number+=1
        
    model.load_state_dict(torch.load(model_path))
    encoder = model.encoder.cuda()
    encoder = encoder.train(False)

    train_dataloader  = DataLoader(cifar_train, batch_size=len(cifar_train), shuffle=True)
    test_dataloader   = DataLoader(cifar_test, batch_size=len(cifar_test), shuffle=True)
    valid_dataloader  = DataLoader(cifar_valid, batch_size=len(cifar_valid), shuffle=True)

    data, _        = next(iter(train_dataloader))
    data           =  data.to(device)

    test_data, test_labels = next(iter(test_dataloader))
    test_data      = test_data.to(device)

    valid_data, _  = next(iter(valid_dataloader))
    valid_data     = valid_data.to(device)

    with torch.no_grad():
        encoded_train = encoder.forward(data)
        encoded_valid = encoder.forward(valid_data)
        encoded_test  = encoder.forward(test_data)

    gis = GIS.GIS(encoded_train, data_validate=encoded_valid, verbose=False)

    torch.save(gis, os.path.join('/global/cscratch1/sd/vboehm/AbatiExperiments', 'cifar_GIS_{}_{}'.format(index,code_length)))

    logps       = gis.evaluate_density(encoded_train).detach().cpu().numpy()
    logps_valid = gis.evaluate_density(encoded_valid).detach().cpu().numpy()
    logps_test  = gis.evaluate_density(encoded_test).detach().cpu().numpy()


    percentile = []
    for ii in range(len(logps_test)):
        percentile.append(scipy.stats.percentileofscore(logps_test,logps_test[ii])/100.)


    auroc          = metrics.roc_auc_score(test_labels, np.asarray(percentile))

    results = pickle.load(open('results_cifar.dict', 'rb'))

    results[index] = auroc

    pickle.dump(results, open('results_cifar.dict', 'wb'))
    print('results updated')

