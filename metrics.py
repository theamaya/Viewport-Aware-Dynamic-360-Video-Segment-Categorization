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
from numpy import linalg as LA
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph
from scipy import spatial
from time import sleep
from tqdm import tqdm 
#tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from scipy.spatial import distance

def percentSphereDiff(features, clusteredPoints): #single video features
  arr=[]
  for i in range(len(clusteredPoints)):
    ar=[]
    for j in clusteredPoints[i]:
      for k in clusteredPoints[i]:
        val= abs(features[j][8]-features[k][8])
        ar.append(val)
    arr.append(np.mean(ar))

  return arr

def speedEval(pitch1arr,pitch2arr ,yaw1arr,yaw2arr):
    '''video= int(features[i][0])
    user= int(features[i][1])
    trace1=[]
    trace1.append(data[video][1][user][start1*10:(start1+length)*10])
    trace1.append(data[video][2][user][start1*10:(start1+length)*10])
    
    start2=starts[math.floor(j/3971)]
    video= int(features[j][0])
    user= int(features[j][1])
    trace2=[]
    trace2.append(data[video][1][user][start2*10:(start2+length)*10])
    trace2.append(data[video][2][user][start2*10:(start2+length)*10])
    '''
    trace1=[pitch1arr,yaw1arr]
    trace2=[pitch2arr,yaw2arr]
    speedtrace1=[]
    speedtrace2=[]
    for k in range(1,3):
        speedtrace1.append(math.sqrt((trace1[0][k*10-1]-trace1[0][(k-1)*10])**2 + (trace1[1][k*10-1]-trace1[1][(k-1)*10])**2))
        speedtrace2.append(math.sqrt((trace2[0][k*10-1]-trace2[0][(k-1)*10])**2 + (trace2[1][k*10-1]-trace2[1][(k-1)*10])**2))
        
    return spatial.distance.euclidean(speedtrace1, speedtrace2)

def geodesicDistancePoints(pa,pb,ywa,ywb):
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

def geodesicDistanceTrajs(pitch1arr,pitch2arr ,yaw1arr,yaw2arr):
  ## for two trajectories
  vals=[]
  for i in range(20):
    pa,ywa,pb,ywb= pitch1arr[i] ,yaw1arr[i] ,pitch2arr[i] ,yaw2arr[i]
    #print([pa,pb,ywa,ywb])
    xa= math.cos(pa)*math.cos(ywa)
    ya= math.cos(pa)*math.sin(ywa)
    za= math.sin(pa)
    xb= math.cos(pb)*math.cos(ywb)
    yb= math.cos(pb)*math.sin(ywb)
    zb= math.sin(pb)
    #print(xa, ya, za, xb, yb, zb)
    try:
      g= math.acos((xa*xb+ya*yb+za*zb)/ math.sqrt((xa*xa+ya*ya+za*za)*(xb*xb+yb*yb+zb*zb)))
      gn=100-g*100/ math.pi
      vals.append(gn)
    except ValueError:
        pass
        #print(xa, ya, za, xb, yb, zb)
    
  return np.mean(vals)


def evalTwoVidChunks3(video1, start1, video2, start2, videoFeatures):

  v1= video1
  v2= video2
  s1= int(start1/2)
  s2= int(start2/2)
  f1= videoFeatures[v1*14+s1][2:]
  f2= videoFeatures[v2*14+s2][2:]
  return distance.euclidean(f1, f2)

def withinVideoCluster(clusteredVideos, videoFeatures):
  vals=[]
  for i in tqdm(range(len(clusteredVideos))):
    print(i)
    q=[]
    for u in range(20000):
      a,b= random.sample(range(0, len(clusteredVideos[i])), 2)
      v1,s1= int(clusteredVideos[i][a][0]), int(clusteredVideos[i][a][1])
      v2,s2= int(clusteredVideos[i][b][0]), int(clusteredVideos[i][b][1])
      q.append(evalTwoVidChunks3(v1, s1, v2, s2, videoFeatures))
    vals.append(q)

  return vals

def betweenVideoClusters(clusteredVideos, videoFeatures):
  between=[]
  for k in tqdm(range(len(clusteredVideos))):
    vals=[]
    for i in range(len(clusteredVideos)):
      if i!=k:
        for u in range(20000):
          a= random.sample(range(0, len(clusteredVideos[k])), 1)[0]
          b= random.sample(range(0, len(clusteredVideos[i])), 1)[0]
          v1,s1= int(clusteredVideos[k][a][0]), int(clusteredVideos[k][a][1])
          v2,s2= int(clusteredVideos[i][b][0]), int(clusteredVideos[i][b][1])
          between.append(evalTwoVidChunks3(v1, s1, v2, s2, videoFeatures))
        #between.append(np.mean(vals))

  return between


def withinVPCluster(i, clusteredPointsMy,data):
  #for ith cluster of clusteredPointsMy
  overl=[]
  speedd=[]
  for j in range(50000):
    a,b= random.sample(range(0, len(clusteredPointsMy[i])), 2)
    video1, user1, start1= int(clusteredPointsMy[i][a][0]), int(clusteredPointsMy[i][a][1]), int(clusteredPointsMy[i][a][2]) 
    video2, user2, start2= int(clusteredPointsMy[i][b][0]), int(clusteredPointsMy[i][b][1]), int(clusteredPointsMy[i][b][2]) 
    pitch1arr= data[video1][1][user1][start1*10:(start1+2)*10]
    pitch2arr= data[video2][1][user2][start2*10:(start2+2)*10]
    yaw1arr= data[video1][2][user1][start1*10:(start1+2)*10]
    yaw2arr= data[video2][2][user2][start2*10:(start2+2)*10]
    overl.append(geodesicDistanceTrajs(pitch1arr,pitch2arr ,yaw1arr,yaw2arr))
    speedd.append(speedEval(pitch1arr,pitch2arr ,yaw1arr,yaw2arr))
    
  return overl, speedd

def crossVPCluster(i, clusteredPointsMy, data):
  overl=[]
  speedd=[]
  for j in range(100000):
    a= random.sample(range(0, len(clusteredPointsMy[i])), 1)[0]
    q=i
    while q==i:
      q= random.sample(range(0, len(clusteredPointsMy)), 1)[0]
    b= random.sample(range(0, len(clusteredPointsMy[q])), 1)[0]
    video1, user1, start1= int(clusteredPointsMy[i][a][0]), int(clusteredPointsMy[i][a][1]), int(clusteredPointsMy[i][a][2]) 
    video2, user2, start2= int(clusteredPointsMy[q][b][0]), int(clusteredPointsMy[q][b][1]), int(clusteredPointsMy[q][b][2]) 
    pitch1arr= data[video1][1][user1][start1*10:(start1+2)*10]
    pitch2arr= data[video2][1][user2][start2*10:(start2+2)*10]
    yaw1arr= data[video1][2][user1][start1*10:(start1+2)*10]
    yaw2arr= data[video2][2][user2][start2*10:(start2+2)*10]
    overl.append(geodesicDistanceTrajs(pitch1arr,pitch2arr ,yaw1arr,yaw2arr))
    speedd.append(speedEval(pitch1arr,pitch2arr ,yaw1arr,yaw2arr))
    
  return overl, speedd