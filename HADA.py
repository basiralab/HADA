#!/usr/bin/python
# -*- coding: utf8 -*-
"""Main function of Hierarchical Adversarial Domain Alignment (HADA) framework 
   for predicting a target graph from a source graph and classifying the subjects 
   using their original source graphs and their predicted target graphs.
    
    Details can be found in:
    (1) the original paper https://link.springer.com/chapter/10.1007/978-3-030-32281-6_11
        Alaa Bessadok, Mohamed Ali Mahjoub, and Islem Rekik. "Hierarchical Adversarial Connectomic Domain Alignment 
        for Target Brain Graph Prediction and Classification From a Source Graph", PRIME-MICCAI workshop 2019, Shenzhen, China.
    (2) the youtube channel of BASIRA Lab: https://www.youtube.com/watch?v=OJOtLy9Xd34

    ---------------------------------------------------------------------
    
    This file contains the implementation of three key steps of our HADA framework:
    (1) hierarchical domain alignment and (2) target graph prediction:
        HADA(sourceGraph,targetGraph,labels,settings)
                Inputs:
                        sourceGraph: (n × m) matrix stacking the source graphs of all subjects
                                     n the total number of subjects
                                     m the number of features
                        targetGraph: (n × m) matrix stacking the target graphs of all subjects
                                     n the total number of subjects
                                     m the number of features
                        labels:      a list of labels such as healthy or disordered subjects
                        settings:    store the neural network settings such as the dimension of the embedded graphs
                                     and the type of autoencoder we choosed (variational or simple autoencoder)
                Output:
                        mae:         mean absolute error (MAE) estimation measuring the prediction error between 
                                     the ground truth graphs and their corresponding predicted graphs
                        dataset_source_and_predicted_target:(n × m) matrix stacking the original source graphs
                                    and the predicted target graphs
                        testlabel:   a list of labels of all testing subjects

    (3) disease classification:
                       RandomForestClassifier: a python function for learning a binary random forest classifier

    To evaluate our framework we used Leave-One-Out crossvalidation strategy.
        
     Sample use:
     mae, dataset_source_and_predicted_target, testlabel = HADA(sourceGraph,targetGraph,labels,settings)
    
    ---------------------------------------------------------------------

    Copyright 2019 Alaa Bessadok, Sousse University.
    Please cite the above paper if you use this code.
    All rights reserved.
    """

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

#from keras.utils import to_categorical
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy import stats
from math import exp

from encoder import Encoder
import tensorflow as tf
import numpy as np
import settings
import SIMLR
import time


start = time.time()


def HADA(sourceGraph, targetGraph, labels, settings):

    # initialisation
    subject = 150
    overallResult_PCC = np.zeros((subject,32))
    overallResult_TSW = np.zeros((subject,32))
    allSV = np.empty((0,sourceGraph.shape[1]), int)
    allTV = np.empty((0,targetGraph.shape[1]), int)
    allpredTV = np.empty((0,targetGraph.shape[1]), int)
    testlabel = []
    # Create training and testing sets
    loo = LeaveOneOut()
    loo.get_n_splits(sourceGraph)
    for train_index, test_index in loo.split(sourceGraph):
        rearrangedPredictorView = np.concatenate((np.transpose(sourceGraph[train_index]), np.transpose(sourceGraph[test_index])),axis = 1)
        rearrangedTargetView = np.concatenate((np.transpose(targetGraph[train_index]),np.transpose(targetGraph[test_index])),axis = 1)
        
    ## Domain Alignment (DA) using ARGA and Similarity matrix learning using SIMLR
        simlr = SIMLR.SIMLR_LARGE(1, 50, 0)
        enc = Encoder(settings)
        
    ## STEP 1: Hierarchical Domain Alignment for traing samples
        print("Hierarchical Domain Alignment for traing samples")
        print("level 1")
        Simlarity2, _,_, _ = simlr.fit(targetGraph[train_index])
        encode_S_T = enc.erun(Simlarity2, sourceGraph[train_index])
        
        # H denotes the number of hierarchical levels
        H = 2
        temporary = encode_S_T
        for number in range (1,H):
            print("level ", H)
            encode_train__TV_A = enc.erun(Simlarity2, temporary)
            temporary = encode_train__TV_A
    
    ## STEP 2: Target Graph Prediction
    ## STEP 2.1: Source graph embedding of training and testing subjects
        test__train__SV = np.vstack((sourceGraph[train_index],sourceGraph[test_index]))
        print("Source graph embedding of training and testing subjects...")
        Simlarity1, _,_, _ = simlr.fit(test__train__SV)
        encode_test__train__SV = enc.erun(Simlarity1, test__train__SV)

    ## STEP 2.2: Connectomic Manifold Learning using SIMLR
        print("SIMLR...")
        SALL, FALL,val, ind = simlr.fit(encode_test__train__SV)
        SY, FY,val, ind = simlr.fit(encode_train__TV_A)
        # number of neighbors for trust score
        TS_bestNb = 5
        # get the best neighbors in the learned manifold of the regularized source graph embeddings
        sall = SALL.todense()
        Index_ALL = np.argsort(-sall, axis=0)
        des = np.sort(-sall, axis=0)
        Bvalue_ALL = -des
        # get the best neighbors in the learned manifold of the hierarchically aligned source and target graph embeddings
        sy = SY.todense()
        Index_Y = np.argsort(-sy, axis=0)
        desy = np.sort(-sy,axis=0)
        Bvalue_Y = -desy
        
        # make prediction for each testing subject
        for testingSubject in range(1,2):
            print "testing subject:", test_index
            # get this testing subject's rearranged index and original index
            tSubjectIndex = (sourceGraph[train_index].shape[0]-2) + testingSubject
            tSubjectOriginalIndex = test_index
            # compute Tscore for each neighbor
            trustScore = np.ones((TS_bestNb,TS_bestNb))
            newWeight_TSW = np.ones(TS_bestNb)

            for neighbor in range(0,TS_bestNb):
                neighborIndex = Index_ALL[tSubjectIndex,neighbor]
                temp_counter = 0
                while (neighborIndex  > sourceGraph[train_index].shape[0]):
                # best neighbor is a testing data 
                    temp_counter = temp_counter + 1
                    neighborIndex = Index_ALL[tSubjectIndex,(TS_bestNb + temp_counter)]

                if (temp_counter != 0):
                    neighborSequence = TS_bestNb + temp_counter
                else:
                    neighborSequence = neighbor

                    #print(neighborIndex)
                    # get top nb neighbors in mappedX
                    neighborListX = Index_ALL[neighborIndex,0:TS_bestNb]
                    # get top nb neighbors in mappedY
                    neighborListY = Index_Y[neighborIndex,0:TS_bestNb]
                    # calculate trust score
                    trustScore[TS_bestNb-1,neighbor] = len(np.intersect1d(np.array(neighborListX),np.array(neighborListY)))
                    # calculate new weight (TS * Similarity)
                    newWeight_TSW[neighbor] = exp(trustScore[TS_bestNb-1,neighbor] / TS_bestNb * Bvalue_ALL[tSubjectIndex,neighborSequence])

            #reconstruct with Tscore and similarity weight
            innerPredict_TSW = np.zeros(sourceGraph[train_index].shape[1])[np.newaxis]
            #summing up the best neighbors
            for j1 in range(0,TS_bestNb):
                tr = (rearrangedTargetView[:,Index_ALL[tSubjectIndex,j1]])[np.newaxis]
                if j1 == 0:
                    innerPredict_TSW = innerPredict_TSW.T + tr.T * newWeight_TSW[j1]
                else:
                    innerPredict_TSW = innerPredict_TSW + tr.T * newWeight_TSW[j1]

            # scale weight to 1
            Scale_TSW = sum(newWeight_TSW)
            innerPredict_TSW = np.divide(innerPredict_TSW, Scale_TSW)
                
            # calculate result (MAE)
            tr2 = (rearrangedTargetView[:,tSubjectIndex])[np.newaxis]
            resulttsw =abs(tr2.T - innerPredict_TSW)
            iMAE_TSW = mean_absolute_error(tr2.T, innerPredict_TSW)
            overallResult_TSW[tSubjectOriginalIndex,TS_bestNb] = overallResult_TSW[tSubjectOriginalIndex,TS_bestNb] + iMAE_TSW
 
            allSV = np.append(allSV, sourceGraph[test_index], axis=0)
            testlabel.append(labels[test_index])
            allpredTV = np.append(allpredTV, innerPredict_TSW.T, axis=0)

            print test_index
            
    dataset_source_and_predicted_target = np.concatenate((allSV, allpredTV),axis=1)
    
    print('END')
    
    mae = np.mean(overallResult_TSW,axis=0)
    print("Mean Absolute Error: ")
    print(mae[np.nonzero(mae)])

    
    return mae, dataset_source_and_predicted_target, testlabel



## INPUT DATA
## Simulate graph data for simply running the code
## in this exemple, the source and target matrices have different statistical distributions
mu, sigma = 0.22266368090882432, 0.027202072213276744 # mean and standard deviation
sourceGraph = np.random.normal(mu, sigma, (150,595))
mu, sigma = 0.08308065685993601, 0.01338490182696101
targetGraph = np.random.normal(mu, sigma, (150,595))
labels = np.concatenate((np.zeros((1,75)),np.ones((1,75))), axis=None)

## HADA execution
model = 'arga_ae' #autoencoder/variational autoencoder
settings = settings.get_settings_new(model)
mae, dataset_source_and_predicted_target, testlabel = HADA(sourceGraph,targetGraph,labels,settings)


## STEP 3: Disease Classification using Random Forest
classes = testlabel
label = np.array(classes)
loo = LeaveOneOut()
actual_label = []
predicted_sv_predtv_label = []

RF = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=None,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

#Random search of parameters, using 3 fold cross validation, search across 100 different combinations, and use all available cores:
#RF = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
for train_index, test_index in loo.split(dataset_source_and_predicted_target):
    
    train_data, test_data, train_labels, test_labels = dataset_source_and_predicted_target[train_index], dataset_source_and_predicted_target[test_index], label[train_index], label[test_index]
    actual_label  = np.append(actual_label, test_labels)
    RF.fit(train_data,np.ravel(train_labels))
    y_pred = RF.predict(test_data)
    predicted_sv_predtv_label = np.append(predicted_sv_predtv_label, y_pred)
    

print("Accuracy score: ")
print(accuracy_score(actual_label, predicted_sv_predtv_label))

end = time.time()
print(end - start)

# -*- coding: utf-8 -*-