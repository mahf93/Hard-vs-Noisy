import scipy.spatial.distance
from scipy.spatial.distance import hamming
from scipy.spatial import distance
import numpy as np
import torch
from utils import *
import torch.nn as nn
import torch.nn.functional as nnf
import pandas as pd

def distance_to_center_features(row):
    f = row['features']
    m = row['mean-features']
    return np.linalg.norm(f - m)

def distance_to_center_features_apad(row):
    f = row['features']
    m = row['mean-features-adaptive']
    return np.linalg.norm(f - m)

def distance_to_center_features_cos_adap(row):
    f = row['features']
    m = row['mean-features-adaptive']
    return scipy.spatial.distance.cosine(f , m)

def distance_to_center_features_cos(row):
    f = row['features']
    m = row['mean-features']
    return scipy.spatial.distance.cosine(f , m)

# function to compute KL Divergence
"""KL Divergence(P|Q)"""
def KLD(p_probs, q_probs): 
    print('here')
    print(p_probs / q_probs)
    print(np.log(p_probs / q_probs))
    
    KLD = p_probs * np.log(p_probs / q_probs)
    return np.sum(KLD)
def KL(a, b): # KL Divergence that deals with zero
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

# function to compute JS Divergence
def compute_JSD(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    # normalize
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (KL(p, m) + KL(q, m)) / 2

def compute_mean_except(arr, label):
    avg = 0
    for i, a in enumerate(arr):
        if i != label:
            avg+=a
    return avg/(len(arr)-1)

def test(epoch, net, testloader, device, criterion):
    net.eval()
    test_loss = 0
    acc1 = 0
    acc5 = 0
    with torch.no_grad():
        for batch_idx, (input_ids, inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            top1, top5 = [accitem.item() for accitem in accuracy(outputs, targets, (1,5))]
            acc1 += top1
            acc5 += top5
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%%'
                         % (test_loss/(batch_idx+1), acc1/(batch_idx+1)))
    
   
    return (test_loss/(batch_idx+1), acc1/(batch_idx+1), acc5/(batch_idx+1))

# Training
def train(epoch, net, trainloader, device, criterion, optimizer, model_name, num_classes):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    acc1 = 0
    acc5 = 0

    # we are saving for each sample the correct class, loss, probability of prediction, class prediction
    dict_infos = {}
    index = 0
    for batch_idx, (input_ids, inputs, targets, or_targets) in enumerate(trainloader):
        inputs, targets, or_targets = inputs.to(device), targets.to(device), or_targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs) 

        # compute features of size 1280
        # this if function should be adjusted if another architecture is used
        if model_name == 'mobilenet_v2':
            features = net.features(inputs)
    
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            # features = nn.AdaptiveAvgPool2d(1)(features)
            features = torch.flatten(features, 1)
        if model_name == 'densenet':
            features = net.features(inputs)
            features = F.relu(features, inplace=True)
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)  
            
        for img_id, feature, tar, out, or_tar in zip(input_ids, features, targets, outputs, or_targets):
            l = criterion(out.unsqueeze(0), tar.unsqueeze(0))
            prob = nnf.softmax(out, dim=0)
            top_p, top_class = prob.topk(2)
            
            # for each sample, store the vector of size 1280 which includes the relu activation
            # ons and offs in the last feature layer
            feature_vector = (feature>0).cpu().numpy()
            
            
            dict_infos[index] = {'id':img_id, 'label':tar.item(), 'original_label':or_tar.item(),
                               'loss':l.item(), 'prediction':top_class[0].item(), 'prediction_probability':top_p[0].item(),
                                 'features': feature_vector, 'prediction_probability_class':prob[tar].item(),
                                 'prediction_probability_max_others':top_p[1].item(), 'prob_vec': prob.cpu().detach().numpy()
                               }
            index += 1
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        top1, top5 = [accitem.item() for accitem in accuracy(outputs, targets, (1,5))]
        acc1 += top1
        acc5 += top5

        progress_bar(batch_idx, len(trainloader), 'Loss: %.2f | Acc: %.2f%%'
                    % (train_loss/(batch_idx+1),  acc1/(batch_idx+1)))
        
    df_infos = pd.DataFrame.from_dict(dict_infos, "index")
    df_infos['features'] = df_infos['features'].apply(lambda x: x.astype(float))
    df_infos['mean-features'] = df_infos.apply(lambda row: np.mean(df_infos[df_infos['label']==row['label']]['features']), axis=1)
    df_infos['SCD'] = df_infos.apply(lambda row: distance_to_center_features(row), axis=1)
    df_infos['one_hot_label'] = df_infos.apply(lambda row: np.array(np.eye(num_classes)[row['label']]),axis=1)
    df_infos['JSD'] = df_infos.apply(lambda row: compute_JSD(row['one_hot_label'], row['prob_vec']), axis=1)
    df_infos['mean-predics'] = df_infos.apply(lambda row: np.mean(df_infos[df_infos['label']==row['label']]['prob_vec']), axis=1)
    df_infos['mean-predics-class'] = df_infos.apply(lambda row: row['mean-predics'][row['label']], axis=1) # Hc
    df_infos['Lc'] = df_infos.apply(lambda row: compute_mean_except(row['mean-predics'], row['label']), axis=1)
    
    df_infos['W_of_JSD'] = df_infos.apply(lambda row: min(row['prediction_probability']/row['prediction_probability_class'], 
                                                         max(row['mean-predics'])/row['mean-predics-class']), axis=1)
    df_infos['mean-features-adaptive'] = df_infos.apply(lambda row: np.mean(df_infos[((df_infos['label']==row['label'])&(df_infos['prediction_probability_class']>=df_infos['Lc']))|
                                                                                     ((df_infos['label']!=row['label'])&(df_infos['prediction']==row['label'])&(df_infos['prediction_probability']>row['mean-predics-class']))]['features']), axis=1)
    df_infos['ACD'] = df_infos.apply(lambda row: distance_to_center_features_cos_adap(row), axis=1)
    
    return (train_loss/(batch_idx+1), acc1/(batch_idx+1), acc5/(batch_idx+1), df_infos[['id', 'label', 'original_label',
                                                                                       'loss', 'prediction', 'prediction_probability',
                                                                                       'prediction_probability_class', 'prediction_probability_max_others',
                                                                                       'JSD', 'W_of_JSD', 'ACD',
                                                                                       'SCD']])  


def train_plain(epoch, net, trainloader, device, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    acc1 = 0
    acc5 = 0

    # we are saving for each sample the correct class, loss, probability of prediction, class prediction
    index = 0
    for batch_idx, (input_ids, inputs, targets, or_targets) in enumerate(trainloader):
        inputs, targets, or_targets = inputs.to(device), targets.to(device), or_targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs) 
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        top1, top5 = [accitem.item() for accitem in accuracy(outputs, targets, (1,5))]
        acc1 += top1
        acc5 += top5

        progress_bar(batch_idx, len(trainloader), 'Loss: %.2f | Acc: %.2f%%'
                    % (train_loss/(batch_idx+1),  acc1/(batch_idx+1)))
    
    
    
    return (train_loss/(batch_idx+1), acc1/(batch_idx+1), acc5/(batch_idx+1))
