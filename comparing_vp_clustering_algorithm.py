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
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import davies_bouldin_score
from sklearn.mixture import GaussianMixture as GMM
import random
from scipy import spatial
from time import sleep
from tqdm import tqdm 
import shutil  
import argparse
import os.path
from os import path

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

def getFeaturesOneVideo(videoNum, start, length=2):
    features= []
    path= folder_path+ 'extracted_features/'+str(start)+'-'+str(length+start)+'s/'
    for i in range(videoNum,videoNum+1):
        f= open(path+str(i)+'.txt',"r")

        arr= f.readlines()
        for j in arr:
            features.append(list(map(float,j.split(' '))))
        f.close()

    features= np.array(features)
    
    return features

def getFeatures(videoNum,start):
    features= []
    #start=30   ##### or 26
    length=2

    path= folder_path+ 'extracted_features/'+str(start)+'-'+str(length+start)+'s/'
    for i in range(videoNum,videoNum+1):#88):
        f= open(path+str(i)+'.txt',"r")

        arr= f.readlines()
        for j in arr:
            features.append(list(map(float,j.split(' '))))
        f.close()

        #features2.append(arr)
    features= np.array(features)
    
    return features

def geodesicDistance(pa,pb,ywa,ywb):
    #print([pa,pb,ywa,ywb])
    xa= math.cos(pa)*math.cos(ywa)
    ya= math.cos(pa)*math.sin(ywa)
    za= math.sin(pa)
    xb= math.cos(pb)*math.cos(ywb)
    yb= math.cos(pb)*math.sin(ywb)
    zb= math.sin(pb)
    #print(xa, ya, za, xb, yb, zb)
    #try:

    g= math.acos((xa*xb+ya*yb+za*zb)/ math.sqrt((xa*xa+ya*ya+za*za)*(xb*xb+yb*yb+zb*zb)))
    #except ValueError:
    #    print('Fail')
        #pass
        #print(xa, ya, za, xb, yb, zb)
    gn=100-g*100/ math.pi
    return gn

def speedEval(pitch1arr,pitch2arr ,yaw1arr,yaw2arr):

    trace1=[pitch1arr,yaw1arr]
    trace2=[pitch2arr,yaw2arr]
    speedtrace1=[]
    speedtrace2=[]
    for k in range(1,3):
        speedtrace1.append(math.sqrt((trace1[0][k*10-1]-trace1[0][(k-1)*10])**2 + (trace1[1][k*10-1]-trace1[1][(k-1)*10])**2))
        speedtrace2.append(math.sqrt((trace2[0][k*10-1]-trace2[0][(k-1)*10])**2 + (trace2[1][k*10-1]-trace2[1][(k-1)*10])**2))
        
    return spatial.distance.euclidean(speedtrace1, speedtrace2)

def percentSphereDiff(features, clusteredPoints):
  arr=[]
  for i in range(len(clusteredPoints)):
    ar=[]
    for j in clusteredPoints[i]:
      for k in clusteredPoints[i]:
        val= abs(features[j][8]-features[k][8])
        ar.append(val)
    arr.append(np.mean(ar))

  return arr

def geoVpOverlap(videoNum, clusteredPoints,start,length=2):
    scores=[]
    overall=[]
    speedScores=[]
    features= getFeatures(videoNum,start)

    for i in range(len(clusteredPoints)):
        if len(clusteredPoints[i])>1:
            collection=[[],[]]
            for j in clusteredPoints[i]:
                collection[0].append(data[videoNum][1][j][start*10:(start+length)*10])
                collection[1].append(data[videoNum][2][j][start*10:(start+length)*10])
            
            myScores=[]
            for k in range(len(collection[0])):
                for j in range(len(collection[0])):
                    if k<j:
                        for p in range(20):
                            val=geodesicDistance(collection[0][k][p],collection[0][j][p], collection[1][k][p],collection[1][j][p])
                            scores.append(val)
                            myScores.append(val)
                        speedScores.append(speedEval(collection[0][k],collection[0][j] ,collection[1][k],collection[1][j]))
                            
                        
            overall.append(min(myScores))
    sphericalPercent= percentSphereDiff(features, clusteredPoints)
    return(np.mean(scores), np.std(scores), np.mean(speedScores), np.std(speedScores), np.mean(sphericalPercent), np.std(sphericalPercent))

def getClusteredPoints(path):
    clusteredPoints=[]
    f= open(path,"r")
    arr= f.readlines()
    for j in arr:
        clusteredPoints.append(list(map(int,j.split(' '))))
    f.close()
    return (clusteredPoints)

def readTraj(video,start):
  path= folder_path+ 'trajectoryClusteringResults/'+str(video)+'/'+str(start)+'_trajSpectral.txt'
  try:
    trajClusters= getClusteredPoints(path)
    return trajClusters
  except:
    return None
  #return trajClusters

def readSpheri(video,start):
  path= folder_path+ 'sphericalClusteringResults/'+str(video)+'/'+str(start)+'.txt'
  try:
    spheriClusters= getClusteredPoints(path)
    return spheriClusters
  except:
    return None
  return spheriClusters

def readDB(video,start):
  path= folder_path+ 'DBClusteringResults/'+str(video)+'/'+str(start)+'.txt'
  try:
    DBClusters= getClusteredPoints(path)
    return DBClusters
  except:
    return None
  #return DBClusters

def myCluster(video,start,n):
    features= getFeaturesOneVideo(video, start, length=2)
    X = np.array(features)
    space= X[:,2:]
    #cols= ['%25 yaw', 'mean yaw', '%75 yaw', '%25 pitch', 'mean pitch', '%75 pitch', '% sphere', 'max move pitch', 'max move yaw', '%25 speed yaw', 'mean speed yaw', '%75 speed mean', '%25 speed pitch', 'mean speed pitch', '%75 speed pitch']
    df = pd.DataFrame(X[:,2:])#, columns= cols)

    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)#, columns = cols)

    kmeans = KMeans(n_clusters=n, random_state=0).fit(df_normalized)
    a=kmeans.labels_
    clusteredMy=[]
    clusteredPointsMy=[]
    for j in range(n):
        clusteredPointsMy.append([])

    for i in range(len(a)):
        ind= a[i]
        clusteredPointsMy[ind].append(int(X[i][1]))  
    clusteredPointsMy.sort(key=len, reverse=True)

    return clusteredPointsMy

def myClusterOptimum(video,start):
    features= getFeaturesOneVideo(video, start, length=2)
    X = np.array(features)
    space= X[:,2:]
    #cols= ['%25 yaw', 'mean yaw', '%75 yaw', '%25 pitch', 'mean pitch', '%75 pitch', '% sphere', 'max move pitch', 'max move yaw', '%25 speed yaw', 'mean speed yaw', '%75 speed mean', '%25 speed pitch', 'mean speed pitch', '%75 speed pitch']
    df = pd.DataFrame(X[:,2:])#, columns= cols)

    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)#, columns = cols)

    scores = []
    centers = list(range(2,len(features)))
    for center in centers:
        #print(center, end=' ')
        scores.append(get_kmeans_score(np.asarray(df_normalized), center))

    opt= scores.index(min(scores))+2

    kmeans = KMeans(n_clusters=opt, random_state=0).fit(df_normalized)
    a=kmeans.labels_
    clusteredMy=[]
    clusteredPointsMy=[]
    for j in range(opt):
        clusteredPointsMy.append([])

    for i in range(len(a)):
        ind= a[i]
        clusteredPointsMy[ind].append(int(X[i][1]))  
    clusteredPointsMy.sort(key=len, reverse=True)

    return clusteredPointsMy

parser = argparse.ArgumentParser()

parser.add_argument("--path", help="the path to the top folder")

args = parser.parse_args()
if args.path:
    folder_path= args.path
else:
    folder_path= 'E:/Internship/academics/Amaya/NOSSDAV2021/Viewport-Aware-Dynamic-360-Video-Segment-Categorization/'

data= importAggregatedDataset()

results=[[[],[],[],[],[],[],[]],[[],[],[],[],[],[],[]],[[],[],[],[],[],[],[]],[[],[],[],[],[],[],[]]]
for video in tqdm(range(88)):
  for start in range(30,32,2):
    tra= readTraj(video,start)
    sp= readSpheri(video,start)
    DB= readDB(video,start)
    if tra!= None and sp!= None and DB!= None:
      my= myCluster(video,start,len(sp))
      a,b,c,d,e,f= geoVpOverlap(video, tra, start,length=2)
      results[0][0].append(a)
      results[0][1].append(b)
      results[0][2].append(c)
      results[0][3].append(d)
      results[0][4].append(e*100)
      results[0][5].append(f)
      results[0][6].append(len(tra))
      a,b,c,d,e,f= geoVpOverlap(video, sp, start,length=2)
      results[1][0].append(a)
      results[1][1].append(b)
      results[1][2].append(c)
      results[1][3].append(d)
      results[1][4].append(e*100)
      results[1][5].append(f)
      results[1][6].append(len(sp))
      a,b,c,d,e,f= geoVpOverlap(video, DB, start,length=2)
      results[2][0].append(a)
      results[2][1].append(b)
      results[2][2].append(c)
      results[2][3].append(d)
      results[2][4].append(e*100)
      results[2][5].append(f)
      results[2][6].append(len(DB))
      a,b,c,d,e,f= geoVpOverlap(video, my, start,length=2)
      results[3][0].append(a)
      results[3][1].append(b)
      results[3][2].append(c)
      results[3][3].append(d)
      results[3][4].append(e*100)
      results[3][5].append(f)
      results[3][6].append(len(my))

#alignment for boxplot
vpOverlap= [results[0][0],results[1][0],results[2][0],results[3][0]]
speedDiff= [results[0][2],results[1][2],results[2][2],results[3][2]]
percentDiff= [results[0][4],results[1][4],results[2][4],results[3][4]]
clusterNums= [results[0][6],results[1][6],results[2][6],results[3][6]]





boxprops_1 = dict(linestyle='--', linewidth=3, color='c')
boxprops_2 = dict(linestyle='--', linewidth=3, color='c')
boxprops_3 = dict(linestyle='-.', linewidth=3, color='m')
boxprops_4 = dict(linestyle='-.', linewidth=3, color='m')


vid = ['T', 'S', 'D', 'P']
color = ['r', 'g', 'b', 'k']
hatch = ['//', '+', 'o', '']
fig, ax = plt.subplots(1,1)
fig.set_figheight(10)
fig.set_figwidth(16)


bp = ax.boxplot(vpOverlap, showfliers=False, patch_artist=True, boxprops=boxprops_1)
for box_ind, box in enumerate(bp['boxes']):
            box.set(color=color[box_ind], linewidth=2)
            box.set(facecolor='#ffffff')
            box.set(hatch=hatch[box_ind])
ax.grid()
ax.set_ylabel('%', fontsize=60)
plt.xticks(np.arange(1, 5), vid)
ax.tick_params(axis='both', labelsize=60)
plt.xticks(np.arange(1, 5), vid)
plt.tight_layout()
plt.savefig(folder_path+'comparing_vp_clustering_algorithm_results/BOX_PLOT_VPalgoComparison-VPoverlap.png')
plt.close()



fig, ax = plt.subplots(1,1)
fig.set_figheight(10)
fig.set_figwidth(16)

bp = ax.boxplot(speedDiff, showfliers=False, patch_artist=True, boxprops=boxprops_1)
for box_ind, box in enumerate(bp['boxes']):
            box.set(color=color[box_ind], linewidth=2)
            box.set(facecolor='#ffffff')
            box.set(hatch=hatch[box_ind])
ax.grid()
ax.set_ylabel('rad/s', fontsize=60)
plt.xticks(np.arange(1, 5), vid)
ax.tick_params(axis='both', labelsize=60)
plt.xticks(np.arange(1, 5), vid)
plt.tight_layout()
plt.savefig(folder_path+'comparing_vp_clustering_algorithm_results/BOX_PLOT_VPalgoComparison-Speeddif.png')
plt.close()


fig, ax = plt.subplots(1,1)
fig.set_figheight(10)
fig.set_figwidth(16)

bp = ax.boxplot(percentDiff, showfliers=False, patch_artist=True, boxprops=boxprops_1)
for box_ind, box in enumerate(bp['boxes']):
            box.set(color=color[box_ind], linewidth=2)
            box.set(facecolor='#ffffff')
            box.set(hatch=hatch[box_ind])
ax.grid()
ax.set_ylabel('%', fontsize=60)
plt.xticks(np.arange(1, 5), vid)
ax.tick_params(axis='both', labelsize=60)
plt.xticks(np.arange(1, 5), vid)
plt.tight_layout()
plt.savefig(folder_path+'comparing_vp_clustering_algorithm_results/BOX_PLOT_VPalgoComparison-%sphere.png')
plt.close()

fig, ax = plt.subplots(1,1)
fig.set_figheight(10)
fig.set_figwidth(16)

bp = ax.boxplot(clusterNums, showfliers=False, patch_artist=True, boxprops=boxprops_1)
for box_ind, box in enumerate(bp['boxes']):
            box.set(color=color[box_ind], linewidth=2)
            box.set(facecolor='#ffffff')
            box.set(hatch=hatch[box_ind])
ax.grid()
ax.set_ylabel('Number of clusters', fontsize=60)
plt.xticks(np.arange(1, 5), vid)
ax.tick_params(axis='both', labelsize=60)
plt.xticks(np.arange(1, 5), vid)
plt.tight_layout()
plt.savefig(folder_path+ 'comparing_vp_clustering_algorithm_results/BOX_PLOT_VPalgoComparison-nclusters.png')
plt.close()