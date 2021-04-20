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
#from tqdm.notebook import tqdm as tqdm
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
  
def normalizeFeatures(features):
    X = np.array(features)
    space= X[:,3:]
    #cols= ['%25 yaw', 'mean yaw', '%75 yaw', '%25 pitch', 'mean pitch', '%75 pitch', '% sphere', 'max move pitch', 'max move yaw', '%25 speed yaw', 'mean speed yaw', '%75 speed mean', '%25 speed pitch', 'mean speed pitch', '%75 speed pitch']
    df = pd.DataFrame(X[:,3:])#, columns= cols)

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

def normalizeFeatures(features):
    X = np.array(features)
    space= X[:,2:]
    #cols= ['%25 yaw', 'mean yaw', '%75 yaw', '%25 pitch', 'mean pitch', '%75 pitch', '% sphere', 'max move pitch', 'max move yaw', '%25 speed yaw', 'mean speed yaw', '%75 speed mean', '%25 speed pitch', 'mean speed pitch', '%75 speed pitch']
    df = pd.DataFrame(X[:,2:])#, columns= cols)

    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)#, columns = cols)

    return df_normalized

# Davies Bouldin score for K means
from sklearn.metrics import davies_bouldin_score
def get_kmeans_score(data, center):
    '''
    returns the kmeans score regarding Davies Bouldin for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the Davies Bouldin score for the kmeans model fit to the data
    '''
    #instantiate kmeans
    kmeans = KMeans(n_clusters=center)
    # Then fit the model to your data using the fit method
    model = kmeans.fit_predict(data)
    
    # Calculate Davies Bouldin score
    score = davies_bouldin_score(data, model)
    
    return score


parser = argparse.ArgumentParser()

parser.add_argument("--path", help="the path to the top folder")
parser.add_argument("--viewportClusters", help="number of viewport clusters. default is 10")

args = parser.parse_args()
if args.path:
    folder_path= args.path
else:
    folder_path= setPath.setFolderPath() 

if args.viewportClusters:
    viewportClusters= int(args.viewportClusters)
else:
    viewportClusters=10

    
print('Making clusters...')
data= importAggregatedDataset()
allFeatures= readAllFeatures()
df_normalized= normalizeFeatures(allFeatures)
hierFeatures= featureHierarchy(allFeatures)
clusteredVPs, df_normalized= clusterMyWay(allFeatures, viewportClusters)
videoFeatures= getVideoFeatures(clusteredVPs, viewportClusters,data)
normalizedVideoF= normalizeFeatures(videoFeatures)

print('Calculating DBScore...')
scores = []
centers = list(range(2,30))
for center in centers:
    scores.append(get_kmeans_score(normalizedVideoF, center))

print('Plotting...')
fig, ax = plt.subplots(1,1, figsize=(16,12))
plt.plot(centers, scores, linestyle='--', marker='o', color='b')
ax.tick_params(axis='both', labelsize=40)
plt.xlabel('Number of video Clusters', fontsize=40)
plt.ylabel('Davies Bouldin score', fontsize=40)
plt.tight_layout()
plt.savefig(folder_path+'optimum_video_categories_number_analysis_results/DBScoreVideoCategories_VP_clusters_'+str(viewportClusters)+'.png')
plt.close()

