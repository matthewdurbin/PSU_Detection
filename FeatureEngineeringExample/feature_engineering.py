# -*- coding: utf-8 -*-
"""
Prelinimary "Feature Engineering" for INMM abstract

This script loads in MCNP datasts of direcitonal detection scenarios
with 4 detector arrays (2x4x16 NaI)
Co, Cs, and Ir - 10,000 trials each from 1-5 m from center

RFC are run for various combinations of input features, and improtances
calculated

Author: Matthew Durbin
"""

# Imports (some may not be needed)

import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn import neighbors
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.utils import shuffle
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold
from sklearn.inspection import permutation_importance
import os

#%% Norm
def norm(inp):
    """
    Takes input (inp) and normalizes it row wise to unity
    """
    out = np.zeros((np.shape(inp)))
    sums = np.sum(inp, axis=1)[:, None]
    for i in range(len(inp)):
        if sums[i] > 0: 
            out[i] = inp[i]/sums[i]
    return(out)

#%% Analysis
def analysis(y, y_p):
    # calculates error, accoutning for circular statistics
    score = accuracy_score(y, y_p)
    error = np.mean(180-abs(abs(y-y_p)-180))
    return(score, error)
#%% Load data
# Format:
# totals (4), peaks (4), compton (4), spectral bins (40), x, y, z, radius (cm), angle (degrees)
data_co = np.load('dataset_Co_tpb.npy')
data_cs = np.load('dataset_Cs_tpb.npy')
data_ir = np.load('dataset_Ir_tpb.npy')
data = np.concatenate((data_co, data_cs, data_ir))
#%% Create Hot Coded Isotope Info
"""
This creates three additional binary input features
Is this Co? y/n (0/1)
Is this Cs? y/n (0/1)
Is this Ir? y/n (0/1)
"""
iso = np.zeros((len(data),3))
iso[:10000,0] = np.ones((10000))
iso[10000:20000,1] = np.ones((10000))
iso[20000:,2] = np.ones((10000))

#%% Assign Data
y = np.round(data[:,-1]) # angle
x_total = norm(data[:,0:4]) # total counts
x_peak = norm(data[:,4:8]) # photo peak counts
x_compton = norm(data[:,8:12]) # compton region counts
x_both = np.column_stack((x_peak, x_compton)) # peak and comptons
x_pci = np.column_stack((x_both, iso)) # peak, compton, and isotope
x_bins = norm(data[:,12:52]) # 10 spectral bins

#%% Create Hot Coded Detector order 
# which of the detectors has the most counts?
det_or = np.zeros((x_total.shape))
for i in range(len(det_or)):
    det_or[i,np.argmax(x_total[i])] = 1
   
#%% Assign more Data   
x_pcd = np.column_stack((x_both, det_or)) # peak, compton, and detector ordder
x_pcid = np.column_stack((x_pci, det_or)) # peak, comton, isotope, and detector order
#%% RFC
def run_rfc(x,y):
    """
    This loads a RFC, and a stratified k-fold cross validaiton scheme
    x are input features, y is the label 
    returns the mean +/- std dev accuracy (ACC) and
    average angular error (AAE) across the k-folds
    """
    rfc = RandomForestClassifier()
    nfold = 5
    skf = StratifiedKFold(n_splits=nfold, shuffle=True)
    accs = np.zeros(nfold)
    aaes = np.zeros(nfold)
    for fno, (tr_index, ts_index) in enumerate(skf.split(x, y)):
            # print("Folding", fno+1, "/", nfold)
            x_tr, y_tr = x[tr_index], y[tr_index]
            x_ts, y_ts = x[ts_index], y[ts_index]
            rfc.fit(x_tr, y_tr)
            accs[fno], aaes[fno] = analysis(y_ts, rfc.predict(x_ts))
            
    aae = [np.mean(aaes),np.std(aaes)]
    acc = [np.mean(accs),np.std(accs)]
    print("ACC: ", acc[0], " +/- ", acc[1])
    print("AAE: ", aae[0], " +/- ", aae[1])
    return (acc, aae)

#%% Expirments - IFs 
# Run a RFC for varios combinations of input features
print("===========================")
print("Totals")
print("===========================")
acc_total, aae_total = run_rfc(x_total, y)

print("===========================")
print("Peaks")
print("===========================")
acc_peak, aae_peak = run_rfc(x_peak, y)

print("===========================")
print("Peaks and Compton")
print("===========================")
acc_both, aae_both = run_rfc(x_both, y)

print("===========================")
print("Peaks, Compton, Iso")
print("===========================")
acc_pci, aae_pci = run_rfc(x_pci, y)

print("===========================")
print("Peaks, Compton, Det Order")
print("===========================")
acc_pcd, aae_pcd = run_rfc(x_pcd, y)

print("===========================")
print("Peaks, Compton, Iso, Det Order")
print("===========================")
acc_pcid, aae_pcid = run_rfc(x_pcid, y)

print("===========================")
print("Bins")
print("===========================")
acc_bins, aae_bins = run_rfc(x_bins, y)
print("===========================")


#%% Feature improtances 
def feature_importance(x,y):
    """
    Calculates the Gini improtance/ mean decrease in importance
    for inputs x and labels y. Returns, and plots, these importances
    """
    rfc = RandomForestClassifier()
    rfc.fit(x, y)
    importances = rfc.feature_importances_
    plt.figure()
    plt.bar(np.arange(0,len(x[0]),1),importances)
    plt.xlabel('Feature Number')
    plt.ylabel('Relative Importance')
    
    return(importances)

#%% Feaure importances expirements
print("===========================")
print("Peaks and Compton")
print("===========================")
fi_both = feature_importance(x_both, y)

print("===========================")
print("Peaks, Compton, Iso")
print("===========================")
fi_pci = feature_importance(x_pci, y)

print("===========================")
print("Peaks, Compton, Det Order")
print("===========================")
fi_pcd = feature_importance(x_pcd, y)

print("===========================")
print("Peaks, Compton, Iso, Det Order")
print("===========================")
fi_pcid = feature_importance(x_pcid, y)

print("===========================")
print("Bins")
print("===========================")
fi_bins = feature_importance(x_bins, y)
print("===========================")

#%%  Permutatio Importance
"""
This type of importance is much more time consuming, so i just give one example
It calculates the Mean Decrease in Accuracy if a feature value is shuffled 
It creates a dictonary with importances over 5 permutations, the means, and std
"""
rfc = RandomForestClassifier()
rfc.fit(x_pcid, y)
p_imp = permutation_importance(rfc, x_pcid, y)
plt.bar(np.arange(0,len(x_pcid[0]),1),p_imp['importances_mean'], yerr=p_imp['importances_std'])
plt.xlabel('Feature Number')
plt.ylabel('Relative Importance')



