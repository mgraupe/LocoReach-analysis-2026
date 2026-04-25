import pdb
import numpy as np
import matplotlib.pyplot as plt
import pickle
import umap
import sklearn.cluster as cluster
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import glob

def optimizeT_SNE(n_comp,X):
    silMax = [0.,-1,-1,-1]
    for i in range(4):
        for j in range(10):
            per = 5 * (i + 1)
            ee = 2 * (j + 1)
            embedding = TSNE(n_components=n_comp, perplexity=per, early_exaggeration=ee, random_state=42)
            clusterable_embedding = embedding.fit_transform(X)
            for n_clusters in range(2,7):
                clusterer = KMeans(n_clusters=n_clusters,init='k-means++',n_init=100)
                cluster_labels = clusterer.fit_predict(clusterable_embedding)

                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed
                # clusters
                silScore = silhouette_score(X, cluster_labels)

                #silScore = metrics.silhouette_score(clusterable_embedding, Y, metric='euclidean')
                if silScore > silMax[0]:
                    silMax[0] = silScore; silMax[1] = per; silMax[2] = ee; silMax[3] = n_clusters
                print(i, j, per, ee, n_clusters, silScore)
    print('max sil score for TSNE : ', silMax)
    return silMax

animals = ['220211_f38','220214_f43','220205_f57','220205_f61','220507_m81','220507_m90','220525_m19','220525_m27','220525_m28', '220716_f65', '220716_f67']

#mliPath = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/simplexSummary/results_final_Heike/'
mliPath = '/home/mgraupe/Downloads/Heike_results/'

paw = 'FL' # chose paw for analysis
#phase = 'off' # for stance onset
cell_type = 'PC-simple'

timeLimits = [-0.2,0.2] # in ms

time = np.linspace(-0.15,0.15,31)
pawDict = {'FL':0,'FR':1,'HL':2,'HR':3}
allModTraces = []
nModTraces = 0
nKernelsTot = 0
modTraceInfo = {}
for m in animals:
    print(m)
    cells = glob.glob(mliPath+m+'/%s_*_significance.p' % cell_type)
    for i in range(len(cells)):
        rec = cells[i][-29:-15]
        #pdb.set_trace()
        [significance_on, significance_off, psth_on, psth_off, kernel_on, kernel_off] = pickle.load(open(mliPath+m+'/'+cell_type+'_'+rec+'_significance.p', 'rb'))
        if significance_off[pawDict[paw]]:
            modTraceInfo[nModTraces] = {}
            allModTraces.append(kernel_off[pawDict[paw]])
            modTraceInfo[nModTraces]['mouse'] = m
            modTraceInfo[nModTraces]['recDate'] = rec
            #modTraceInfo[nModTraces]['recDateNb'] = mliProfiles[1]['recDateNb']
            #pdb.set_trace()
            nModTraces += 1
        nKernelsTot+=1

print('%s out ot %s kernels are significant for %s ' % (nModTraces,nKernelsTot,paw))
arrAllModTrac = np.asarray(allModTraces)

X = np.copy(arrAllModTrac)
Y = np.ones(len(X))

print(np.shape(X))
#pdb.set_trace()
n_comp = 2
clusterAlgorithms = {
    #'UMAP': umap.UMAP(
    #        n_neighbors=45,
    #        min_dist=0.0,
    #        n_components=n_comp,
    #        random_state=42,),
    'Isomap': Isomap(n_neighbors=6,n_components=n_comp),
    'T-SNE': TSNE(n_components=n_comp,perplexity=5,early_exaggeration=2,random_state=42),
}

fig = plt.figure(figsize=(12,20))
plt.figtext(0.08, 0.98, 'MLI PSTH profiles', clip_on=False, color='black',  size=16)
plt.subplots_adjust(left=0.07, right=0.96, top=0.95, bottom=0.08)
#plt.figtext(0.555, 0.96, 'B', clip_on=False, color='black', size=22)
#fig.text()
nFig = 1
clusterDict = {}
clusterDict['classes'] = Y
clusterDict['DesignMatrix'] = X
cc = ['C0','C1','C2','C3','C4','C5','C6','C7','C8']
#clusterDict['ephysDict'] = ephysDict
#clusterDict['nCellsRecordings'] = [nCells, nRCount]
for key in clusterAlgorithms:
    print(key)
    ax0 = fig.add_subplot(4, 2, nFig)
    ax1 = fig.add_subplot(4,2,nFig+2)
    ax3 = fig.add_subplot(4,2,nFig+6)
    ax1.axhline(y=0,ls='--',c='0.6')
    ax1.axvline(x=0, ls='--', c='0.6')
    ax2 = fig.add_subplot(4, 2, nFig + 4)
    ax2.axhline(y=0, ls='--', c='0.6')
    ax2.axvline(x=0, ls='--', c='0.6')
    ax0.set_title(key)
    # ax = plt.figure().add_subplot(projection='3d')
    #if key == 'T-SNE':
    #    t_sne_params = optimizeT_SNE(n_comp,X)
    #    print('optimized params. :', t_sne_params)
    #    clusterAlgorithms[key] = TSNE(n_components=n_comp,perplexity=t_sne_params[2],early_exaggeration=t_sne_params[1],random_state=42)
    #    print(clusterAlgorithms[key])
    clusterable_embedding = clusterAlgorithms[key].fit_transform(X)
    #silScore = metrics.silhouette_score(clusterable_embedding, Y, metric='euclidean')
    #ax0.set_title(key + ': Silhouette score = ' + str(silScore))

    #print(key + ': Silhouette Coefficient = ' + str(silScore))
    # plt.show()
    for n_clusters in range(2,3):
        clusterer = KMeans(n_clusters=n_clusters,init='k-means++',n_init=100)
        cluster_labels = clusterer.fit_predict(clusterable_embedding)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )
    ax0.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], c=cluster_labels, s=10, cmap='Spectral')
    nRCount = 0
    # pdb.set_trace()
    clus1avg = []
    clus2avg = []
    for n in range(len(X)):
        #nRecs = len(ephysDict[n]['mouse-cell-recs'][3])
        #for i in range(nRecs):
        ttt = str(nRCount) + ' ' + modTraceInfo[n]['mouse'] + ' ' + modTraceInfo[n]['recDate']
        # print(ttt)
        ax0.annotate(ttt, (clusterable_embedding[nRCount, 0], clusterable_embedding[nRCount, 1]), alpha=0.2, size=6)
        nRCount += 1
        if cluster_labels[n]==0:
            ax1.plot(time,X[n],lw=0.5,c=cc[cluster_labels[n]],alpha=0.5)
        elif cluster_labels[n]==1:
            ax2.plot(time, X[n],lw=0.5, c=cc[cluster_labels[n]],alpha=0.5)
    #pdb.set_trace()
    mmiI = np.argmin(X,axis=1)
    mmaI = np.argmax(X,axis=1)
    mmiT = time[mmiI]
    mmiA = np.take_along_axis(X, mmiI[:,None], axis=1)
    mmaT = time[mmaI]
    mmaA = np.take_along_axis(X, mmaI[:,None], axis=1)
    clus1avg = np.average(X[cluster_labels==0],axis=0)
    clus2avg = np.average(X[cluster_labels == 1], axis=0)
    ax1.plot(time,clus1avg,c=cc[0])
    ax2.plot(time, clus1avg, c=cc[0])
    ax2.plot(time, clus2avg,c=cc[1])
    ax1.plot(time, clus2avg, c=cc[1])
    bb = np.linspace(-0.155, 0.155, 32)
    if np.max(clus1avg)>np.max(clus2avg):
        ax1.plot(mmaT[cluster_labels==0],mmaA[cluster_labels==0],'o',ms=0.6,c=cc[0])
        ax2.plot(mmiT[cluster_labels==1],mmiA[cluster_labels==1],'o',ms=0.6,c=cc[1])
        ax3.plot(mmaT[cluster_labels == 0], mmaA[cluster_labels == 0], 'o', ms=0.6, c=cc[0])
        ax3.plot(mmiT[cluster_labels == 1], -mmiA[cluster_labels == 1], 'o', ms=0.6, c=cc[1])
        ax3.hist(mmaT[cluster_labels == 0], bins=bb, color=cc[0], histtype='step')
        ax3.hist(mmiT[cluster_labels == 1], bins=bb, color=cc[1], histtype='step')
    else:
        ax2.plot(mmaT[cluster_labels==1],mmaA[cluster_labels==1],'o',ms=0.6,c=cc[1])
        ax1.plot(mmiT[cluster_labels==0],mmiA[cluster_labels==0],'o',ms=0.6,c=cc[0])
        ax3.plot(mmaT[cluster_labels == 1], mmaA[cluster_labels == 1], 'o', ms=0.6, c=cc[1])
        ax3.plot(mmiT[cluster_labels == 0], -mmiA[cluster_labels == 0], 'o', ms=0.6, c=cc[0])
        ax3.hist(mmiT[cluster_labels == 0], bins=bb, color=cc[0], histtype='step')
        ax3.hist(mmaT[cluster_labels == 1], bins=bb, color=cc[1], histtype='step')
    nFig += 1


    #ax3.plot(mmaT[cluster_labels == 0], mmaA[cluster_labels == 0], 'o', ms=0.6, c=cc[0])
    #ax3.plot(mmiT[cluster_labels == 1], mmiA[cluster_labels == 1], 'o', ms=0.6, c=cc[1])

    #ax3.hist(mmaT[cluster_labels==0],bins=bb,color=cc[0],histtype='step')
    #ax3.hist(mmiT[cluster_labels==1],bins=bb,color=cc[1],histtype='step')
    #clusterDict[key] = {}
    #clusterDict[key]['clusterable_embedding'] = clusterable_embedding
    del clusterable_embedding, clusterer
print(time)
print(bb)
#plt.show()
print('shape of all sig. modulated traces :', np.shape(arrAllModTrac))
plt.savefig('PC-simple-kernel-profiles_%s-%s.pdf' % (paw,phase))

#pdb.set_trace()

