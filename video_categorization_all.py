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
#import datasetScripts as ds
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
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import pandas as pd
from sklearn import preprocessing

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#%matplotlib widget
#%config InlineBackend.figure_format='retina'
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
from tqdm.notebook import tqdm as tqdm
from sklearn.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from scipy import spatial

import datetime
import argparse
import os.path
from os import path
from tqdm import tqdm
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
  
# def normalizeFeatures(features):
#     X = np.array(features)
#     space= X[:,3:]
#     #cols= ['%25 yaw', 'mean yaw', '%75 yaw', '%25 pitch', 'mean pitch', '%75 pitch', '% sphere', 'max move pitch', 'max move yaw', '%25 speed yaw', 'mean speed yaw', '%75 speed mean', '%25 speed pitch', 'mean speed pitch', '%75 speed pitch']
#     df = pd.DataFrame(X[:,3:])#, columns= cols)

#     min_max_scaler = preprocessing.MinMaxScaler()
#     np_scaled = min_max_scaler.fit_transform(df)
#     df_normalized = pd.DataFrame(np_scaled)#, columns = cols)

#     return df_normalized


def clusterMyWay(features, nclusters):
    X = np.array(features)
    space= X[:,3:]
    #cols= ['%25 yaw', 'mean yaw', '%75 yaw', '%25 pitch', 'mean pitch', '%75 pitch', '% sphere', 'max move pitch', 'max move yaw', '%25 speed yaw', 'mean speed yaw', '%75 speed mean', '%25 speed pitch', 'mean speed pitch', '%75 speed pitch']
    df = pd.DataFrame(X[:,3:])#, columns= cols)

    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)#, columns = cols)

    n=nclusters
    kmeans = KMeans(n_clusters=n, random_state=0).fit(df_normalized)
    a=kmeans.labels_
    #centers= kmeans.cluster_centers_
    clusteredMy=[]
    clusteredPointsMy=[]
    for j in range(n):
        #clusteredMy.append([])
        clusteredPointsMy.append([])

    for i in range(len(a)):
        ind= a[i]
        #clusteredMy[ind].append(X[i][:2])
        clusteredPointsMy[ind].append(X[i][:3])  
    #kmeans.cluster_centers_
    #centers= getClusterCenters(features, clusteredPointsMy)
    return clusteredPointsMy, df_normalized, #centers

def getVideoFeatures(clusteredPoints, nVPClusters,data):
  videoFeatures=[]

  for i in range(88):
    a=[]
    for j in range(14):
      b=[i,j*2]
      for k in range(nVPClusters):
          b.append(0)
      a.append(b)
    videoFeatures.append(a)
          
  for i in range(len(clusteredPoints)):
      for j in clusteredPoints[i]:
          video= int(j[0])
          time= int(j[2]/2)
          videoFeatures[video][time][i+2]+=1 /len(data[video][1])

  final= []
  for i in range(88):
    for j in videoFeatures[i]:
      final.append(j)

  return np.asarray(final)

def clusterVideos(videoFeatures, nclusters):
  df = pd.DataFrame(videoFeatures[:,2:])

  min_max_scaler = preprocessing.MinMaxScaler()
  np_scaled = min_max_scaler.fit_transform(df)
  df_normalized = pd.DataFrame(np_scaled)

  n=nclusters
  kmeans = KMeans(n_clusters=n, random_state=0).fit(df)
  a=kmeans.labels_

  clusteredVideos=[]
  for i in range(n):
    clusteredVideos.append([])

  for i in range(len(a)):
    ind= a[i]
    clusteredVideos[ind].append(videoFeatures[i][:2])

  return clusteredVideos

def videoClusterLabels(clusteredVideos):
  dynamicVideoLabels=[]
  for i in range(88):
    a=[]
    for j in range(0,28,2):
      a.append(0)
    dynamicVideoLabels.append(a)

  for i in range(len(clusteredVideos)):
    for j in clusteredVideos[i]:
      video, start= int(j[0]), int(j[1]/2)
      dynamicVideoLabels[video][start]=i
  return dynamicVideoLabels






def dotHeatmap(i, videoClustered, path):
  #for the ith cluster
  ys= []
  ps= []
  for k in videoClustered[i]:
    video= int(k[0])
    start= int(k[1])
    for j in range(len(data[video][2])):
      ys.extend(data[video][2][j][start*10:(start+2)*10]) 
      ps.extend(data[video][1][j][start*10:(start+2)*10]) 


  fig, ax = plt.subplots(1,1)#, figsize=(16,12))
  fig.set_figheight(8)
  fig.set_figwidth(16)
  ax.tick_params(labelsize=50)
  plt.hist2d(ys,ps,bins=200)
  fig.tight_layout()
  plt.savefig(path+'/videoCategory_'+str(i)+'_heatmap.png')
  plt.close()

def featureHist(i, videoClustered, videoFeatures, path, viewportClusters ):
    bars=[]
    for j in range(viewportClusters):
      bars.append([])
    for k in videoClustered[i]:
        video= int(k[0])
        start= int(k[1]/2)
        for j in range(viewportClusters):
            bars[j].append(videoFeatures[video*14+start][j+2])  ##

    x_pos=np.arange(viewportClusters)
    fig,ax1 =plt.subplots(1,1)
    fig.set_figheight(8)
    fig.set_figwidth(16)
    ax1.tick_params(labelsize=40)
    plt.ylabel('Formalized feature size',fontsize=40)
    plt.xlabel('Video Category', fontsize=40)
    ax1.yaxis.grid(True)  

    for j in range(viewportClusters):
      if j%2==0:
        ax1.bar(x_pos[j], np.mean(bars[j]), yerr=np.std(bars[j]), align='center', alpha=0.5, ecolor='black', capsize=16,label = x_pos[0], color = 'r')
      else:
        ax1.bar(x_pos[j], np.mean(bars[j]), yerr=np.std(bars[j]), align='center', alpha=0.5, ecolor='black', capsize=16,label = x_pos[1], color = 'g')

    # ax1.bar(x_pos[0], np.mean(bars[0]), yerr=np.std(bars[0]), align='center', alpha=0.5, ecolor='black', capsize=16,label = x_pos[0], color = 'r')
    # ax1.bar(x_pos[1], np.mean(bars[1]), yerr=np.std(bars[1]), align='center', alpha=0.5, ecolor='black', capsize=16,label = x_pos[1], color = 'g')
    # ax1.bar(x_pos[2], np.mean(bars[2]), yerr=np.std(bars[2]), align='center', alpha=0.5, ecolor='black', capsize=16,label = x_pos[0], color = 'r')
    # ax1.bar(x_pos[3], np.mean(bars[3]), yerr=np.std(bars[3]), align='center', alpha=0.5, ecolor='black', capsize=16,label = x_pos[1], color = 'g')
    # ax1.bar(x_pos[4], np.mean(bars[4]), yerr=np.std(bars[4]), align='center', alpha=0.5, ecolor='black', capsize=16,label = x_pos[0], color = 'r')
    # ax1.bar(x_pos[5], np.mean(bars[5]), yerr=np.std(bars[5]), align='center', alpha=0.5, ecolor='black', capsize=16,label = x_pos[1], color = 'g')
    # ax1.bar(x_pos[6], np.mean(bars[6]), yerr=np.std(bars[6]), align='center', alpha=0.5, ecolor='black', capsize=16,label = x_pos[0], color = 'r')
    # ax1.bar(x_pos[7], np.mean(bars[7]), yerr=np.std(bars[7]), align='center', alpha=0.5, ecolor='black', capsize=16,label = x_pos[1], color = 'g')
    # ax1.bar(x_pos[8], np.mean(bars[8]), yerr=np.std(bars[8]), align='center', alpha=0.5, ecolor='black', capsize=16,label = x_pos[0], color = 'r')
    # ax1.bar(x_pos[9], np.mean(bars[9]), yerr=np.std(bars[9]), align='center', alpha=0.5, ecolor='black', capsize=16,label = x_pos[1], color = 'g')

    fig.tight_layout()
    plt.savefig(path+'/VideoCategoryFeatures'+str(i)+'.png')
    plt.close()

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['fliers'], color=color)

parser = argparse.ArgumentParser()

parser.add_argument("--path", help="the path to the top folder")
parser.add_argument("--viewportClusters", help="number of viewport clusters. default is 10")
parser.add_argument("--videoCats", help="number of video categories. default is 6")
parser.add_argument("--saveClusters", action='store_true', help="save the elemets of resulting categories to text files")
parser.add_argument("--savePlots", action='store_true', help="save the plots in figure 9")
parser.add_argument("--eval", action='store_true', help="evaluate and save the plots in figure 10 b")

args = parser.parse_args()
if args.path:
    folder_path= args.path
else:
    folder_path= setPath.setFolderPath() 

if args.viewportClusters:
    viewportClusters= int(args.viewportClusters)
else:
    viewportClusters=10

if args.videoCats:
    videoCats= int(args.videoCats)
else:
    videoCats=6

print('Creating clusters and categorizing videos...')
data= importAggregatedDataset()
allFeatures= readAllFeatures()
#df_normalized= normalizeFeatures(allFeatures)
hierFeatures= featureHierarchy(allFeatures)
clusteredVPs, df_normalized= clusterMyWay(allFeatures, viewportClusters)
videoFeatures= getVideoFeatures(clusteredVPs, viewportClusters,data)
clusteredVideos= clusterVideos(videoFeatures, videoCats)
dynamicVideoLabels = videoClusterLabels(clusteredVideos)

if args.saveClusters or args.savePlots or args.eval:
    if not path.exists(folder_path+'video_categorization_all_results/VPclusters_'+str(viewportClusters)+'_videocats_'+str(videoCats)):
         os.mkdir(folder_path+'video_categorization_all_results/VPclusters_'+str(viewportClusters)+'_videocats_'+str(videoCats))


if args.savePlots:
    print('Saving the plots...')
    path= folder_path+'video_categorization_all_results/VPclusters_'+str(viewportClusters)+'_videocats_'+str(videoCats)
    for i in tqdm(range(videoCats)):
        dotHeatmap(i, clusteredVideos, path)
        featureHist(i, clusteredVideos, videoFeatures, path, viewportClusters )


if args.saveClusters:
    print('Saving clusters...')
    path= folder_path+'video_categorization_all_results/VPclusters_'+str(viewportClusters)+'_videocats_'+str(videoCats)
    for i in tqdm(range(videoCats)):
        f= open(path+'/videoCategory_'+str(i)+'.txt',"w")
        for j in range(len(clusteredVideos[i])):
            str1 = ' '.join(str(int(e)) for e in clusteredVideos[i][j])
            f.write(str1+"\n" )
        f.close()

if args.eval:
    path= folder_path+'video_categorization_all_results/VPclusters_'+str(viewportClusters)+'_videocats_'+str(videoCats)
    sim_within_vid_cls= metrics.withinVideoCluster(clusteredVideos, videoFeatures)
    sim_between_vid_cls= metrics.betweenVideoClusters(clusteredVideos, videoFeatures)
    between2=[]
    for i in range(videoCats):
        between2.append(np.mean(sim_between_vid_cls[i*(videoCats-1):i*(videoCats-1)+videoCats]))


    boxprops_1 = dict(linestyle='--', linewidth=3, color='c')
    boxprops_2 = dict(linestyle='--', linewidth=3, color='c')
    boxprops_3 = dict(linestyle='-.', linewidth=3, color='m')
    boxprops_4 = dict(linestyle='-.', linewidth=3, color='m')


    vid = ['T', 'S', 'D', 'P']
    colors = ['r', 'g', 'b', 'k','c','m']
    hatch = ['//', '+', 'o', '','//', '+']


    font = {
                'size': 60,
                }  
    ticks = []
    for i in range(videoCats):
        ticks.append(str(i+1))
    fig, ax1 = plt.subplots(1,1)
    fig.set_figheight(15)
    fig.set_figwidth(20)
    ax1.tick_params(axis = 'both', labelsize= 60)
    color = 'black'
    ax1.set_xlabel('video chunk category', fontsize=60)
    ax1.set_ylabel('Pairwise dist. within category', color=color, fontsize=60)
    ax1.tick_params(axis='y', labelcolor=color)
    positions1= np.arange(videoCats)*3-0.4
    positions2= np.arange(videoCats)*3+0.4
    positions= np.arange(videoCats)*3
    #print(positions1)
    c='red'
    d= 'blue'
    flierprops1 = dict(marker='o', markerfacecolor=c, markersize=3,
                    linestyle='none', markeredgecolor=c)
    flierprops2 = dict(marker='o', markerfacecolor=d, markersize=3,
                    linestyle='none', markeredgecolor=d)

    bp = ax1.boxplot(sim_within_vid_cls, showfliers=False, patch_artist=True, boxprops=boxprops_1)

    for box_ind, box in enumerate(bp['boxes']):
                # change outline color
                box.set(color=colors[box_ind%6], linewidth=2)
                # change fill color
                box.set(facecolor='#ffffff')
                # change hatch
                box.set(hatch=hatch[box_ind%6])

    q=[]
    for i in range(videoCats):
        q.append(i+1)
    ax1.scatter(q, between2, c='b', label= 'cross-category distance',marker='X', s=400)
    ax1.legend(loc='upper right',prop={'size': 40} )
    plt.xticks(range(1,videoCats+1,1), ticks)
    fig.tight_layout()
    plt.grid()
    plt.savefig(path+ '/VideoCatEvaluation_videoCats_'+str(videoCats)+'.png')
    plt.close()