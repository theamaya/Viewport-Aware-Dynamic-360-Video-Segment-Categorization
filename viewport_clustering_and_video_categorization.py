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
import cv2
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import davies_bouldin_score
from sklearn.mixture import GaussianMixture as GMM
import random
from scipy.stats.kde import gaussian_kde
from numpy import linspace
from sklearn.manifold import TSNE
from tabulate import tabulate
from matplotlib.figure import figaspect
#from tqdm.notebook import tqdm as tqdm
from sklearn.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from scipy import spatial
import datetime

#arguments
# 0.folder_path- the top folder Viewport-Aware-Dynamic-360-Video-Segment-Categorization
# 1.videos - default is all videos
# 2.evaluations- default is True
# 3.write_results- default is True  

folder_path= 'E:/Internship/academics/Amaya/NOSSDAV2021/Viewport-Aware-Dynamic-360-Video-Segment-Categorization'
videos= []
evaluation= True
write_results= True
viewport_clusters= 10
video_categories= 6

def selectVideos():
  videos=[0,1,2,3,4,5]

def importAggregatedDataset():
    data=[]
    path= folder_path+ '/AggregatedDataset/'
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
    df = pd.DataFrame(features[:,3:])#, columns= cols)

    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)#, columns = cols)

    return df_normalized


def clusterMyWay(features, nclusters):
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
    return clusteredPointsMy, df_normalized

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

data= importAggregatedDataset()
allFeatures= readAllFeatures()
df_normalized= normalizeFeatures(allFeatures)
hierFeatures= featureHierarchy(allFeatures)
clusteredVPs, df_normalized= clusterMyWay(allFeatures, 10)
videoFeatures= getVideoFeatures(clusteredVPs, 10,data)
clusteredVideos= clusterVideos(videoFeatures, 6)
dynamicVideoLabels = videoClusterLabels(clusteredVideos)

if evaluation:
  withinVPs=[]
  withinSpeeds=[]
  crossVPs=[]
  crossSpeeds=[]
  for i in range(len(clusteredVPs)):
    a,b= metrics.withinVPCluster(0, clusteredVPs, data)
    withinVPs.append(a)
    withinSpeeds.append(b)

  for i in range(len(clusteredVPs)):
    a,b= metrics.crossVPCluster(i, clusteredVPs, data)
    crossVPs.append(a)
    crossSpeeds.append(b)

  sim_within_vid_cls= metrics.withinVideoCluster(clusteredVideos, videoFeatures)
  sim_between_vid_cls= metrics.betweenVideoClusters(clusteredVideos, videoFeatures)

if write_results:
  result_id= datetime.datetime.now().strftime('%Y%m%d%H%M%S')
  os.mkdir(folder_path+'/video_categorization_'+result_id)

  path= folder_path+'/'+result_id+'/details.txt'
  f= open(path,"w")
  f.write('Number of viewport clusters= '+str(viewport_clusters)+"\n" )
  f.write('Number of video segment categories= '+str(video_categories)+"\n" )
  f.close()

  path= folder_path+'/'+result_id+'/evaluations.txt'
  f= open(path,"w")
  f.write('Number of viewport clusters= '+str(viewport_clusters)+"\n" )
  f.write('Number of video segment categories= '+str(video_categories)+"\n" )
  f.close()

  path= folder_path+'/'+result_id+'/video_features.txt'
  f= open(path,"w")
  for i in range(len(videos)):
    f.write('Number of viewport clusters= '+str(viewport_clusters)+"\n" )
  f.close()

  for i in range(viewport_clusters):
    path= folder_path+'/'+result_id+'/viewport_cluster_'+str(i)+'.txt'
    f= open(path,"w")
    f.write('Number of viewport clusters= '+str(viewport_clusters)+"\n" )
    f.write('Number of video segment categories= '+str(video_categories)+"\n" )
    f.close()

  for i in range(video_categories):
    path= folder_path+'/'+result_id+'/video_cluster_'+str(i)+'.txt'
    f= open(path,"w")
    f.write('Number of viewport clusters= '+str(viewport_clusters)+"\n" )
    f.write('Number of video segment categories= '+str(video_categories)+"\n" )
    f.close()
