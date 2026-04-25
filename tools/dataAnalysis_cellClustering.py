import time
import numpy as np
import sys
import os
import scipy, scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile as tiff
from scipy import io
import pdb
import scipy.ndimage
import itertools
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
import cv2
from scipy import signal
from scipy.signal import find_peaks
import pickle
import random
from statsmodels.stats.anova import anova_single
import scikits.bootstrap as boot
from scipy import ndimage
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib
import multiprocessing as mp
from joblib import Parallel, delayed
import scipy.stats as stats
from tools.pyqtgraph.Qt import QtGui, QtCore
import tools.pyqtgraph as pg
matplotlib.use('TkAgg') # WxAgg
from array import array
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import array as arr
from numpy import trapz
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
numcores = mp.cpu_count()-1
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from scipy.stats import vonmises_line
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
import umap
import sklearn.cluster as cluster
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn import metrics

def clusterCellTypes(ephysDict,folder):
    matplotlib.use('TkAgg')
    nCells = len(ephysDict)
    nRCount = 0
    frmax = 117.873444384234360
    dmax = 0.00085
    dScaling = 4.
    parameterOptimization = False
    print(nCells, ' cells in total!')
    # build design matrix and possible classes
    X = []
    Y = []

    for n in range(nCells):
        print(ephysDict[n]['mouse-cell-recs'], ephysDict[n]['cellType'],'visually guided : ', ephysDict[n]['visuallyGuided'])
        nRecs = len(ephysDict[n]['mouse-cell-recs'][3])
        for i in range(nRecs):
            #print('  ', ephysDict[n][i]['recording'],end='')
            #wave = ephysDict[n][i]['ss_wave']
            #print(len(wave),ephysDict[n][i]['ss_avgSpikeParams'][2])
            ac = ephysDict[n][i]['ss_xprob'][~np.isnan(ephysDict[n][i]['ss_xprob'])]
            #pdb.set_trace()
            #acNorm = StandardScaler().fit(ac)
            #tt = ephysDict[n][i]['ss_xprob_span']
            #pdb.set_trace()
            acNorm = 0.5*ac[30:70]/np.mean(np.concatenate((ac[:11],ac[-11:])))
            #acNorm = ac/np.max(ac)
            wave = ephysDict[n][i]['ss_wave'][~np.isnan(ephysDict[n][i]['ss_wave'])]
            waveS = ephysDict[n][i]['ss_wave_span'][~np.isnan(ephysDict[n][i]['ss_wave'])]
            #pdb.set_trace()
            #waveNorm = 2.*(wave - np.min(wave))/(np.max(wave)-np.min(wave)) - 1.
            waveNorm = (-.5)*wave[16:81]/np.min(wave)
            xsingle = acNorm
            xsingle = np.append(xsingle,waveNorm)
            #if (frmax < ephysDict[n][i]['ss_fr']):
            #    frmax = ephysDict[n][i]['ss_fr']
            xsingle = np.append(xsingle,ephysDict[n][i]['ss_fr']/frmax)
            #if dmax < ephysDict[n][i]['ss_avgSpikeParams'][0]:
            #    dmax = ephysDict[n][i]['ss_avgSpikeParams'][0]
            xsingle = np.append(xsingle,dScaling*ephysDict[n][i]['ss_avgSpikeParams'][0]/dmax)
            xsingle = np.append(xsingle,ephysDict[n][i]['ss_spike-count_CVs'])
            xsingle = np.append(xsingle,[1 if ephysDict[n][i]['cs_number']>0 else 0])
            print(ephysDict[n][i]['cs_number'])
            X.append(xsingle)
            Y.extend([0 if ephysDict[n]['cellType']=='MLI' else 1])
            nRCount+=1
    print('max firing rate :', frmax)
    print('max delay : ', dmax)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print(np.shape(X),np.shape(Y))
    #pdb.set_trace()
    #pca = PCA(n_components=10)
    #pca.fit(X)
    #print(pca.explained_variance_ratio_)
    n_comp = 2
    clusterAlgorithms = {
        'UMAP': umap.UMAP(
                n_neighbors=45,
                min_dist=0.0,
                n_components=n_comp,
                random_state=42,),
        'Isomap': Isomap(n_components=n_comp),
        'T-SNE': TSNE(n_components=n_comp,perplexity=45,early_exaggeration=18,random_state=41),
    }

    #tsne = TSNE(n_components)
    #tsne_result = tsne.fit_transform(X)
    #clusterable_embedding = umap.UMAP(
    #    #n_neighbors=20,
    #    #min_dist=0.0,
    #    #n_components=2,
    #    random_state=42,).fit_transform(X)
    #standard_embedding = umap.UMAP(random_state=42).fit_transform(X)
    # parameter optimization
    if parameterOptimization:
        silMax = [0.,0.,0.]
        for i in range(10):
            for j in range(10):
                n_neig = 5*(i+1)
                min_d = (0.+j*0.1)
                embedding = umap.UMAP(
                    n_neighbors=n_neig,
                    min_dist=min_d,
                    n_components=n_comp,
                    random_state=42,)
                clusterable_embedding = embedding.fit_transform(X)
                silScore = metrics.silhouette_score(clusterable_embedding, Y, metric='euclidean')
                if silScore>silMax[0] : silMax[0] = silScore; silMax[1]=n_neig; silMax[2]=min_d
                print(i,j,n_neig,min_d,silScore)
        print('max sil score for UMAP : ', silMax)
        silMax = [0., 0., 0.]
        for i in range(10):
            for j in range(10):
                per = 5 * (i + 1)
                ee = 2*(j +1)
                embedding = TSNE(n_components=n_comp,perplexity=per,early_exaggeration=ee,random_state=42)
                clusterable_embedding = embedding.fit_transform(X)
                silScore = metrics.silhouette_score(clusterable_embedding, Y, metric='euclidean')
                if silScore > silMax[0]: silMax[0] = silScore; silMax[1] = per; silMax[2] = ee
                print(i, j, per, ee, silScore)
        print('max sil score for TSNE : ', silMax)
        pdb.set_trace()
    ## figure
    fig = plt.figure()
    nFig = 1
    clusterDict = {}
    clusterDict['classes'] = Y
    clusterDict['DesignMatrix'] = X
    clusterDict['ephysDict'] = ephysDict
    clusterDict['nCellsRecordings'] = [nCells, nRCount]
    for key in clusterAlgorithms:
        print(key)
        ax0 = fig.add_subplot(1,3,nFig)
        ax0.set_title(key)
        #ax = plt.figure().add_subplot(projection='3d')
        clusterable_embedding = clusterAlgorithms[key].fit_transform(X)
        silScore = metrics.silhouette_score(clusterable_embedding, Y, metric='euclidean')
        ax0.set_title(key + ': Silhouette score = ' + str(silScore))
        ax0.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], c=Y, s=10, cmap='Spectral')
        print(key + ': Silhouette Coefficient = ' + str(silScore))
        #plt.show()
        nRCount=0
        #pdb.set_trace()
        for n in range(nCells):
            nRecs = len(ephysDict[n]['mouse-cell-recs'][3])
            for i in range(nRecs):
                ttt = str(nRCount) + ' ' + ephysDict[n]['cellType'] + ' ' + ephysDict[n]['mouse-cell-recs'][0][-3:]+' '+ephysDict[n]['mouse-cell-recs'][1] + ' rec:' + ephysDict[n]['mouse-cell-recs'][3][i][-3:]
                #print(ttt)
                ax0.annotate(ttt,(clusterable_embedding[nRCount, 0],clusterable_embedding[nRCount, 1]), alpha=0.2,size=6)
                nRCount+=1
        nFig+=1
        clusterDict[key] = {}
        clusterDict[key]['clusterable_embedding'] = clusterable_embedding
    #plt.show()
    #pdb.set_trace()
    # perform PCA on UMAP components
    #pca = PCA(n_components=2)
    #pca.fit(clusterable_embedding)
    #umapPCA = pca.transform(clusterable_embedding)

    #ax1 = fig.add_subplot(132)
    #ax1.scatter(umapPCA[:,0],umapPCA[:,1],c=Y,s=10,cmap='Spectral')

    #plt.show()
    # ax2 = fig.add_subplot(122)
    # kmeans_labels = cluster.KMeans(n_clusters=2).fit_predict(X)
    # ax2.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], c=kmeans_labels, s=10, cmap='Spectral')
    # nRCount=0
    # for n in range(nCells):
    #     nRecs = len(ephysDict[n]['mouse-cell-recs'][3])
    #     for i in range(nRecs):
    #         ttt = ephysDict[n]['cellType'] + ' ' + ephysDict[n]['mouse-cell-recs'][0][-3:]+' '+ephysDict[n]['mouse-cell-recs'][1] + ' rec:' + ephysDict[n]['mouse-cell-recs'][3][i][-3:]
    #         #print(ttt)
    #         ax2.annotate(ttt,(clusterable_embedding[nRCount, 0],clusterable_embedding[nRCount, 1]), alpha=0.2,size=6)
    #         nRCount+=1
    plt.show()
    pdb.set_trace()

    umapFile = 'clusterSummaryData.p'
    pickle.dump(clusterDict, open(folder+umapFile, 'wb'))
    #neigh = KNeighborsClassifier(n_neighbors=3)
    #neigh.fit(X, Y)
    #for
    #    clf.fit(X_train, y_train)
    #pdb.set_trace()

    #pdb.set_trace()


