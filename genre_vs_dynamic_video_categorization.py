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
#from tqdm.notebook import tqdm as tqdm
from sklearn.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from scipy import spatial
import metrics as metrics
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



parser = argparse.ArgumentParser()

parser.add_argument("--path", help="the path to the top folder")
parser.add_argument("--viewportClusters", help="number of viewport clusters. default is 10")
parser.add_argument("--videoCats", help="number of video categories. default is 6")

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
hierFeatures= featureHierarchy(allFeatures)
clusteredVPs, df_normalized= clusterMyWay(allFeatures, viewportClusters)
videoFeatures= getVideoFeatures(clusteredVPs, viewportClusters,data)
clusteredVideos= clusterVideos(videoFeatures, videoCats)
dynamicVideoLabels = videoClusterLabels(clusteredVideos)

genreCat=[7,0,0,6,5,6,3,5,5,7,4,1,6,9,2,1,8,8,8,8,8,5,8,3,7,6,8,3,6,5,8,8,3,2,8,3,2,8,3,2,8,3,2,8,3,2,8,3,2,8,3,2,2,8,8,8,2,2,2,2,6,6,3,3,0,2,6,6,4,6,1,4,6,6,5,6,6,6,3,6,6,6,6,6,8,5,9,8]
#len(genreCat)
genreClustered=[[],[],[],[],[],[],[],[],[],[]]
for i in range(88):
  for j in range(0,28,2):
    genreClustered[genreCat[i]].append([i,j])

#genreClustered

print('evaluating and saving plots...')
path= folder_path+'genre_vs_dynamic_video_categorization_results/'
sim_within_vid_cls= metrics.withinVideoCluster(clusteredVideos, videoFeatures)
sim_between_vid_cls= metrics.betweenVideoClusters(clusteredVideos, videoFeatures)
# between2=[]
# for i in range(videoCats):
#     between2.append(np.mean(sim_between_vid_cls[i*(videoCats-1):i*(videoCats-1)+videoCats]))

genreResults= metrics.withinVideoCluster(genreClustered, videoFeatures)
genreBetween= metrics.betweenVideoClusters(genreClustered, videoFeatures)

resultsAll=[]
for i in range(len(sim_within_vid_cls)):
  resultsAll.extend(sim_within_vid_cls[i])

genreResultsAll=[]
for i in range(len(genreResults)):
  genreResultsAll.extend(genreResults[i])

x=list(np.arange(0,50,2))
fig, ax = plt.subplots(1,1)#, figsize=(16,12))
fig.set_figheight(10)
fig.set_figwidth(16)

plt.subplots_adjust(right= 1,top=1)
font = {
            'size': 50,
            }
font2 = {
            'size': 50,
            }

ax.tick_params(labelsize=60)
plt.ylabel('CDF',font)
plt.xlabel('Pairwise feature distance', font)
ax.grid()
a,=plt.plot(np.sort(genreResultsAll), np.linspace(0, 1, len(genreResultsAll), endpoint=False), 'b',label='Genre- within C.',linewidth=5.0)
b,=plt.plot(np.sort(resultsAll), np.linspace(0, 1, len(resultsAll), endpoint=False),'r', label='Dyn- within C.',linewidth=5.0)
c,=plt.plot(np.sort(sim_between_vid_cls), np.linspace(0, 1, len(sim_between_vid_cls), endpoint=False),'r--', label='Dyn- cross C.',linewidth=5.0)
d,=plt.plot(np.sort(genreBetween), np.linspace(0, 1, len(genreBetween), endpoint=False), 'b--',label='Genre- cross C.',linewidth=5.0)

plt.legend(handles=[a,b,c,d], prop={'size': 40})
fig.tight_layout()
#plt.savefig('E:/Internship/academics/Amaya/FOVanalysis/Analysis/plots/10/'+'Sperical comparison VP overlap - timelapse.pdf')
plt.savefig(path+'genre_vs_dynamic_VPclusters_'+str(viewportClusters)+'_videoCats_'+str(videoCats)+'_.png')
plt.close()