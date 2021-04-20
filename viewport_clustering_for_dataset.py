#%pylab inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image

from scipy.stats.kde import gaussian_kde
from scipy.interpolate import interp1d
from numpy import linspace
import pandas as pd
import metrics as metrics
import statistics
from shapely.geometry import Polygon, Point
from shapely.geometry import box
from shapely.ops import unary_union

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import pandas as pd
from sklearn import preprocessing

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from itertools import chain
#import cv2
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import davies_bouldin_score
from sklearn.mixture import GaussianMixture as GMM
import random
from scipy.stats.kde import gaussian_kde
from numpy import linspace
from sklearn.manifold import TSNE
#from tabulate import tabulate
from matplotlib.figure import figaspect
from tqdm import tqdm
from sklearn.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from scipy import spatial
import datetime
import argparse
import os.path
from os import path
import setPath as setPath

def importAggregatedDataset():
    data=[]
    path= folder_path+ 'aggregated_dataset/'
    for i in range(88):
        
        video=[[],[],[]]
        newpath=path+str(i)+'.txt'
        f= open(newpath,"r")
        arr= f.readlines()
        video[0].append(list(map(float,arr[0].split(' '))))

        for j in range(1,len(arr)):
            if j%2==0:
                video[2].append(list(map(float,arr[j].split(' '))))
            else:
                video[1].append(list(map(float,arr[j].split(' '))))
        f.close()
        data.append(video)

    return data
    

def readAllFeatures():
  path= folder_path+'/extracted_features/allFeatures.txt'
  features=[]
  f= open(path ,"r")

  arr= f.readlines()
  for j in arr:
      a= list(map(float,j.split(' ')))
      features.append(a)
  f.close()

  return (features)


def featureHierarchy(allFeatures):
  hierFeatures=[]
  for i in range(len(data)):
    a=[]
    for j in range(len(data[i][1])):
      a.append([])
    hierFeatures.append(a)

  for i in range(len(allFeatures)):
    video= int(allFeatures[i][0])
    user= int(allFeatures[i][1])
    try:
     hierFeatures[video][user].append(allFeatures[i])
    except:
      print(allFeatures[i])

  return hierFeatures
  
def normalizeFeatures(features):
    X = np.array(features)
    space= X[:,3:]
    #cols= ['%25 yaw', 'mean yaw', '%75 yaw', '%25 pitch', 'mean pitch', '%75 pitch', '% sphere', 'max move pitch', 'max move yaw', '%25 speed yaw', 'mean speed yaw', '%75 speed mean', '%25 speed pitch', 'mean speed pitch', '%75 speed pitch']
    df = pd.DataFrame(X[:,3:])#, columns= cols)

    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)#, columns = cols)

    return df_normalized

def getClusterCenters(hierFeatures, clusteredPoints):
  centers=[]
  for i in range(len(clusteredPoints)):
    arr=[]
    for j in range(len(clusteredPoints[i])):
      video= int(clusteredPoints[i][j][0])
      user= int(clusteredPoints[i][j][1])
      start= int(clusteredPoints[i][j][2]/2)
      arr.append(hierFeatures[video][user][start])
    centers.append(np.mean(arr, axis=0)[3:])
  return centers

def clusterViewports(features, nclusters):
    X = np.array(features)
    space= X[:,3:]
    df = pd.DataFrame(X[:,3:])

    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)

    n=nclusters
    kmeans = KMeans(n_clusters=n, random_state=0).fit(df_normalized)
    a=kmeans.labels_
    clusteredMy=[]
    clusteredPointsMy=[]
    for j in range(n):
        clusteredPointsMy.append([])

    for i in range(len(a)):
        ind= a[i]
        clusteredPointsMy[ind].append(X[i][:3]) 
    centers=[]
    centers= getClusterCenters(hierFeatures, clusteredPointsMy)

    return clusteredPointsMy, df_normalized, centers

def clusterFeatures(i, clusteredVPs):
  arr=[[],[],[],[],[]]
  for k in clusteredVPs[i]:
    video= int(k[0])
    user= int(k[1])
    start= int(k[2]/2)
    try:
        arr[0].append(hierFeatures[video][user][start][1+3])
        arr[1].append(hierFeatures[video][user][start][4+3])
        arr[2].append(abs(hierFeatures[video][user][start][10+3]))
        arr[3].append(abs(hierFeatures[video][user][start][13+3]))
        arr[4].append(hierFeatures[video][user][start][6+3])
    except:
        pass
  return arr

def plotBars(bar,std,x_pos,i, path):
    font = {
        'size': 16,
        }
    fig,(ax1,ax2,ax3) =plt.subplots(1,3,figsize=(35,12))
    plt.subplots_adjust(left=0.13 ,right=0.95 ,wspace=0.6,bottom=0.08 ,top=0.95)
    
    # ax1.set_xticklabels(['',''],fontdict=font)
    # ax2.set_xticklabels(['',''],fontdict=font)
    # ax3.set_xticklabels([''],fontdict=font)
    ax1.tick_params(axis = 'y', labelsize= 60)
    ax2.tick_params(axis = 'y', labelsize= 60)
    ax3.tick_params(axis = 'y', labelsize= 60)
    ax1.bar(x_pos[0], bar[0], yerr=std[0], align='center', alpha=0.5, ecolor='black', capsize=16,label = x_pos[0], color = 'r')
    ax1.bar(x_pos[1], bar[1], yerr=std[1], align='center', alpha=0.5, ecolor='black', capsize=16,label = x_pos[1], color = 'g')
    ax1.set_ylabel("Rad",fontsize=60 )
    ax2.bar(x_pos[2], bar[2], yerr=std[2], align='center', alpha=0.5, ecolor='black', capsize=16,label = x_pos[2], color = 'b')
    ax2.bar(x_pos[3], bar[3], yerr=std[3], align='center', alpha=0.5, ecolor='black', capsize=16,label = x_pos[3], color = 'c')
    ax2.set_ylabel("Rad/s",fontsize=60 ) 
    ax3.bar(x_pos[4], bar[4], yerr=std[4], align='center', alpha=0.5, ecolor='black', capsize=16,label = x_pos[4], color = 'y', width=[2])
    ax3.set_ylabel("Percentage",fontsize=60 )
    
    ax1.legend(loc='lower right',prop={'size': 40})
    ax2.legend(loc='lower right',prop={'size': 40})
    ax3.legend(loc='lower right',prop={'size': 40})
    fig.tight_layout()
    plt.savefig(path+'features_viewport_cluster'+str(i)+'.png')
    plt.close()
    return

def plotclusters(start,clusteredPoints, centers, path, length=2):
    
    for k in range(len(clusteredPoints)):
      #print('Cluster is '+str(k))
      fig, ax = plt.subplots(1,1)#, figsize=(16,12))
      fig.set_figheight(8)
      fig.set_figwidth(16)
      ax.tick_params(labelsize=60)
      for j in range(int(len(clusteredPoints[k])/10)):
        #if cluster_labels[j]==k:
        yaws=[]
        pitches=[]
        video= int(clusteredPoints[k][j][0])
        user= int(clusteredPoints[k][j][1])
        start= int(clusteredPoints[k][j][2])
        yaws.extend(data[video][2][user][start*10:(start+length)*10])
        pitches.extend(data[video][1][user][start*10:(start+length)*10])

        plt.scatter(yaws, pitches, marker='.')
        plt.xlim(-3.14, 3.14)
        plt.ylim(-1.57, 1.57)

      plt.ylabel('Lattitude - rad',fontsize=50)
      plt.xlabel('Longitude - rad', fontsize=50)
      fig.tight_layout()
      plt.savefig(path+ '/VPClusterScatterplot'+str(k)+'.png')
      #plt.show()
      plt.close()
      #print('Mean yaw '+str(centers[k][1]), 'Mean pitch '+str(centers[k][4]), 'Mean speed yaw '+str(centers[k][10]), 'Mean speed pitch '+str(centers[k][13]), '% sphere '+str(centers[k][6]),   )

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['fliers'], color=color)

parser = argparse.ArgumentParser()

parser.add_argument("--path", help="the path to the top folder")
parser.add_argument("--nclusters", help="number of viewport clusters. default is 10")
parser.add_argument("--saveClusters", action='store_true', help="save the elemets of resulting clusters to text files")
parser.add_argument("--savePlots", action='store_true', help="save the plots in figure 7")
parser.add_argument("--eval", action='store_true', help="evaluate and save the plots in figure 6")

args = parser.parse_args()
if args.path:
    folder_path= args.path
else:
    folder_path= setPath.setFolderPath() 

if args.nclusters:
    nclusters= int(args.nclusters)
else:
    nclusters=10
    
print('Forming viewport clusters')
data= importAggregatedDataset()
allFeatures= readAllFeatures()
hierFeatures= featureHierarchy(allFeatures)
clusteredVPs, df_normalized, centers= clusterViewports(allFeatures, nclusters)


if args.saveClusters or args.savePlots or args.eval:
    #print('true')
    if not path.exists(folder_path+'viewport_clustering_for_dataset_results/nclusters_'+str(nclusters)):
         os.mkdir(folder_path+'viewport_clustering_for_dataset_results/nclusters_'+str(nclusters))


if args.saveClusters:
    print('Saving clusters...')
    path= folder_path+'viewport_clustering_for_dataset_results/nclusters_'+str(nclusters)+'/'
    for i in tqdm(range(nclusters)):
        f= open(path+'cluster_'+str(i)+'.txt',"w")
        for j in range(len(clusteredVPs[i])):
            str1 = ' '.join(str(int(e)) for e in clusteredVPs[i][j])
            f.write(str1+"\n" )
        f.close()

if args.savePlots:
    print('Saving plots')
    path= folder_path+'viewport_clustering_for_dataset_results/nclusters_'+str(nclusters)+'/'
    for i in tqdm(range(nclusters)):
        barPlot= clusterFeatures(i, clusteredVPs)
        bar=[np.mean(barPlot[0]), np.mean(barPlot[1]), np.mean(barPlot[2]), np.mean(barPlot[3]), np.mean(barPlot[4])]
        std=[np.std(barPlot[0]), np.std(barPlot[1]), np.std(barPlot[2]), np.std(barPlot[3]), np.std(barPlot[4])]
        x_pos = ['Position - Yaw','Position - Pitch','Speed - Yaw','Speed - Pitch', '% sphere']
        plotBars(bar,std,x_pos,i, path)

    plotclusters(10,clusteredVPs,centers,path,length=2)

if args.eval:
    
    withinVPs=[]
    withinSpeeds=[]
    crossVPs=[]
    crossSpeeds=[]
    print('Evaluating within-cluster metrics...')
    for i in tqdm(range(len(clusteredVPs))):
        a,b= metrics.withinVPCluster(0, clusteredVPs, data)
        withinVPs.append(a)
        withinSpeeds.append(b)
    print('Evaluating cross-cluster metrics...')
    for i in tqdm(range(len(clusteredVPs))):
        a,b= metrics.crossVPCluster(i, clusteredVPs, data)
        crossVPs.append(a)
        crossSpeeds.append(b)

    path= folder_path+'viewport_clustering_for_dataset_results/nclusters_'+str(nclusters)+'/'
    VPBetween=[]
    SpeedDiffBetween=[]
    
    for i in range(nclusters):
        VPBetween.append(np.mean(crossVPs[i]))
        SpeedDiffBetween.append(np.mean(crossSpeeds))

    print('saving results...')
    font = {
            'size': 60,
            } 

    ticks = []
    for i in range(nclusters):
        ticks.append(str(i+1))
    fig, ax1 = plt.subplots(1,1)
    fig.set_figheight(15)
    fig.set_figwidth(30)
    ax1.tick_params(axis = 'both', labelsize= 60)
    ax1.set_ylim((0,100))
    color = 'red'
    ax1.set_xlabel('Cluster', fontsize=60)
    ax1.set_ylabel('% Pairwise Viewport overlap', color=color, fontsize=50)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylim((0,10))
    ax2.tick_params(axis = 'both', labelsize= 60)
    color = 'blue'
    ax2.set_ylabel('Pairwise difference in speed rad/s', color=color, fontsize=50)
    ax2.tick_params(axis='y', labelcolor=color)
        
        
    positions1= np.arange(nclusters)*3-0.4
    positions2= np.arange(nclusters)*3+0.4
    positions= np.arange(nclusters)*3
    c='red'
    d= 'blue'
    flierprops1 = dict(marker='o', markerfacecolor=c, markersize=3,
                    linestyle='none', markeredgecolor=c)
    flierprops2 = dict(marker='o', markerfacecolor=d, markersize=3,
                    linestyle='none', markeredgecolor=d)

    ax1.boxplot(withinVPs,showfliers=True, positions=positions1, boxprops=dict(color=c),
                capprops=dict(color=c),
                whiskerprops=dict(color=c),
                medianprops=dict(color=c),flierprops=flierprops1)
    ax1.scatter(positions1, VPBetween, c=c, label= 'cross-cluster VPO',marker='X', s=400)

    ax2.boxplot(withinSpeeds,showfliers=True, positions=positions2, boxprops=dict(color=d),
                capprops=dict(color=d),
                whiskerprops=dict(color=d),
                medianprops=dict(color=d), flierprops=flierprops2)

    ax2.scatter(positions2, SpeedDiffBetween, c=d, label= 'cross-cluster Speed diff.', marker='X',s=400)
    ax1.legend(loc='upper right',prop={'size': 40} )
    ax2.legend(loc='lower right',prop={'size': 40} )
    plt.xticks(range(0, len(ticks) * 3, 3), ticks)
    fig.tight_layout()
    plt.grid()
    plt.savefig(path+'/VPclusterEval_nclusters_'+str(nclusters)+'.png')
    plt.close()
    