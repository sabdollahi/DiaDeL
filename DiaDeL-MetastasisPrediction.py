import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from torch.autograd import Variable
import torch
import math
import numpy as np

class DiaDeL(nn.Sequential):
    def __init__(self, x_dim):
        super(DiaDeL, self).__init__()
        self.linear1 = nn.Linear(x_dim, 64)
        self.bn1 = nn.BatchNorm1d(num_features = 64)
        self.relu1 =  nn.ReLU()
        self.d1 = nn.Dropout(0.05)
        self.linear2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(num_features = 128)
        self.relu2 =  nn.ReLU()
        self.d2 = nn.Dropout(0.05)
        self.linear3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(num_features = 256)
        self.relu3 = nn.ReLU()
        self.d3 = nn.Dropout(0.05)
        self.linear4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(num_features = 128)
        self.relu4 = nn.ReLU()
        self.d4 = nn.Dropout(0.05)
        self.linear5 = nn.Linear(128, 32)
        self.bn5 = nn.BatchNorm1d(num_features = 32)
        self.relu5 = nn.ReLU()
        self.d5 = nn.Dropout(0.05)
        self.linear6 = nn.Linear(32, 2)
         
    def forward(self,x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.d1(x)
        
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.d2(x)
        
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.d3(x)
        
        x = self.linear4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.d4(x)
        
        x = self.linear5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.d5(x)
        
        x = self.linear6(x)
        return x
#Each Cancer Organ Prediction
main_folder = "MutSigPPI/"
tcga_signatures_dataframe = pickle.load(open(main_folder + "TCGA_30signatures_dataframe.pickle","rb"))
tcga_clinical_dataframe = pickle.load(open(main_folder + "TCGA_clinical_dataframe.pickle","rb"))

#Predict Metastasis (Stage IV) or not (Stages I, II, and III)
#tcga_clinical_dataframe[tcga_clinical_dataframe['stage'] == 'Stage IVA']
which_clinicals = ['stage']
tcga_clinical_dataframe = tcga_clinical_dataframe[which_clinicals]
replace_statement = {}
metastasis_list = ['Stage IV','Stage IVA','Stage IVB','Stage IVC']
other_list = ['Stage I','Stage IA','Stage IB','Stage II','Stage IIA','Stage IIB','Stage IIC','Stage III','Stage IIIA','Stage IIIB','Stage IIIC']
#Metastasis Stage
for m in metastasis_list:
    replace_statement[m] = 1
#Non-metastasis Stage
for o in other_list:
    replace_statement[o] = 0

metastasis_patients = tcga_clinical_dataframe[tcga_clinical_dataframe["stage"].isin(metastasis_list)]
metastasis_patients = metastasis_patients.replace({'stage': replace_statement})
other_patients = tcga_clinical_dataframe[tcga_clinical_dataframe["stage"].isin(other_list)]
other_patients = other_patients.replace({'stage': replace_statement})

start_and_end_for_other = [0,793,1586,2379,3172,3965,4758,5554]
for i in range(7):
    print("PART: " + str(i))
    selected_other_patients = other_patients[start_and_end_for_other[i]:start_and_end_for_other[i+1]]
    folds_accuracy = []
    K = 10 #Kfold (number of parts = K)
    kf_other = KFold(n_splits = K, shuffle = True)
    kf_metastasis = KFold(n_splits = K, shuffle = True)
    parts_metastasis = kf_metastasis.split(metastasis_patients)
    parts_other = kf_other.split(selected_other_patients)
    indices_metastasis = next(parts_metastasis, None)
    indices_other = next(parts_other, None)
    fold_number = 1
    while(indices_metastasis):
        lr = 0.5
        #Define the model
        model = DiaDeL(30)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        batch_size = 20
        for shuffled_epoch in range(10):
            print("Shuffled Epoch= " + str(shuffled_epoch))
            training = metastasis_patients.iloc[indices_metastasis[0]]
            training_other = selected_other_patients.iloc[indices_other[0]]
            training = shuffle(training.append(training_other))
            Y = training[['stage']].values
            Y = Variable(torch.LongTensor(list(Y.flatten())), requires_grad=False)
            training = training.index
            for epoch in range(50):
                for index in range(0, len(training), batch_size):
                    y = Y[index : index + batch_size]
                    batch_X = []
                    for patient in training[index : index + batch_size]:
                        p_data = tcga_signatures_dataframe.loc[patient].tolist()
                        p_data = [int(i) for i in p_data]
                        batch_X.append(p_data)
                    X = torch.FloatTensor(batch_X)
                    optimizer.zero_grad()
                    Y_hat = model(X)
                    loss = criterion(Y_hat, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
        test = metastasis_patients.iloc[indices_metastasis[1]]
        test_other = selected_other_patients.iloc[indices_other[1]]
        test = shuffle(test.append(test_other))
        Y_test = test[['stage']].values
        Y_test = Variable(torch.LongTensor(list(Y_test.flatten())), requires_grad=False)
        test = test.index
        test_list = []
        for patient in test:
            p_data = tcga_signatures_dataframe.loc[patient].tolist()
            p_data = [int(i) for i in p_data]
            test_list.append(p_data)
        test_list = torch.FloatTensor(test_list)
        test_batch_Y_hat = model.forward(test_list)
        dummy, preds_test = torch.max (test_batch_Y_hat, dim = 1)
        accuracy_test = (preds_test == Y_test).long().sum().float() /  preds_test.size()[0]
        print("Fold: " + str(fold_number) + " ACC: " + str(accuracy_test))
        fold_number += 1
        folds_accuracy.append(accuracy_test)
        indices_metastasis = next(parts_metastasis, None)
        indices_other = next(parts_other, None)    
    print("-----------------------------")
