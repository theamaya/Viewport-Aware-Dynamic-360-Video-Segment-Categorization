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
args = parser.parse_args()
if args.path:
    folder_path= args.path
else:
    folder_path= setPath.setFolderPath() 

print('Calculating DBScore...')
data= importAggregatedDataset()
allFeatures= readAllFeatures()
df_normalized= normalizeFeatures(allFeatures)
scores = []
#centers = list(range(2,30))
for center in tqdm(range(2,30)):
    #print(center, end=' ')
    scores.append(get_kmeans_score(np.asarray(df_normalized), center))
 

fig, ax = plt.subplots(1,1, figsize=(16,12))
plt.plot(centers, scores, linestyle='--', marker='o', color='b', linewidth= 4);
ax.tick_params(axis='both', labelsize=50)
plt.xlabel('Number of VP Clusters', fontsize=50);
plt.ylabel('Davies Bouldin score', fontsize=50);
plt.tight_layout()
plt.savefig(folder_path+'optimum_VP_cluster_number_analysis_results/DBscoreVPClusters.png')
plt.close()