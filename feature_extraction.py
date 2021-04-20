import random
import math
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import sys
from numpy import linspace
import statistics
from shapely.geometry import Polygon, Point
from shapely.geometry import box
from shapely.ops import unary_union
from itertools import chain
from sklearn import preprocessing
from tqdm import tqdm

import argparse
import setPath as setPath


def importAggregatedDataset():
    data=[]
    path= folder_path+'aggregated_dataset/'
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
    
def yawPitchStat(i,j, start, length):
    #i th video
    #j th user
    pitch= data[i][1][j][start*10:(start+length)*10]
    yaw= data[i][2][j][start*10:(start+length)*10]
    
    switchArr=[0]
    for i in range(19):
        if abs(yaw[i+1]-yaw[i])>4:
            switchArr.append(1)
        else:
            switchArr.append(0)
            
    switchArr2=[0]        
    for i in range(19):
        switchArr2.append((switchArr[i+1]+switchArr2[i])%2)
    newYaw=[]
    for i in range(20):
        if switchArr2[i]==1:
            if yaw[i]>0:
                newYaw.append(yaw[i]-6.28)
            else:
                newYaw.append(yaw[i]+6.28)
        else:
            newYaw.append(yaw[i])
            
    pmean= np.percentile(pitch,50)
    ymean= np.mean(newYaw)
    p25= np.percentile(pitch,25)
    y25= np.percentile(newYaw,25)
    p75= np.percentile(pitch,75)
    y75= np.percentile(newYaw,75)
    pstd=np.std(pitch)
    ystd=np.std(newYaw)
    
    if ymean<-3.14:
        ymean=6.28+ymean
    elif ymean>3.14:
        ymean= ymean-6.28
        
    if y25<-3.14:
        y25=6.28+y25
    elif y25>3.14:
        y25= y25-6.28
        
    if y75<-3.14:
        y75=6.28+y75
    elif y75>3.14:
        y75= y75-6.28

    return (y25, ymean, y75, p25, pmean, p75)

def getRect(x,y):
    rects=[]
    xplus= x+0.8726
    yplus= y+0.8726
    xminus= x-0.8726
    yminus= y-0.8726
    if xminus<-3.14:
        if yminus<-1.57:
            rects.append(box(xminus+6.28,-1.57, 3.14, yplus))
            rects.append(box(-3.14, -1.57 , xplus, yplus))
            rects.append(box(xminus+6.28,yminus+3.14, 3.14, 1.57))
            rects.append(box(-3.14, yminus+3.14 , xplus, 1.57))

        if yplus>1.57:
            rects.append(box(-3.14,yminus,xplus,1.57))
            rects.append(box(xminus+6.28,yminus,3.14,1.57))
            rects.append(box(-3.14,-1.57,xplus,yplus-3.14))
            rects.append(box(xminus+6.28,-1.57,3.14,yplus-3.14))

        else:
            rects.append(box(-3.14,yminus, xplus, yplus))
            rects.append(box(xminus+6.28, yminus , 3.14, yplus))

    elif xplus>3.14:
        if yminus<-1.57:
            rects.append(box(xminus,-1.57,3.14,yplus))
            rects.append(box(-3.14,-1.57,xplus-6.28,yplus))
            rects.append(box(xminus,yminus+3.14,3.14,1.57))
            rects.append(box(-3.14,yminus+3.14,xplus-6.28,1.57))

        if yplus>1.57:
            rects.append(box(xminus,yminus,3.14,1.57))
            rects.append(box(xminus,-1.57,3.14,yplus-3.14))
            rects.append(box(-3.14,yminus,xplus-6.28,1.57))
            rects.append(box(-3.14,-1.57,xplus-6.28,yplus-3.14))

        else:
            rects.append(box(xminus,yminus, 3.14, yplus))
            rects.append(box(-3.14, yminus , xplus-6.28, yplus))

    elif yplus>1.57:
        rects.append(box(xminus,yminus, xplus, 1.57))
        rects.append(box(xminus, -1.57 , xplus, yplus-3.14))

    elif yminus<-1.57:
        rects.append(box(xminus,-1.57, xplus, yplus))
        rects.append(box(xminus, yminus+3.14 , xplus, 1.57))
    else:
        rects.append(box(xminus,yminus, xplus, yplus))

    return rects

def percentageSphereVideo(i,j, start, length):
    #i th video
    #j th user

    polygon= unary_union(getRect(data[i][2][j][start*10],data[i][1][j][start*10]))
    arr=[]
    for k in range(start*10+1,start*10+length*10,1):
        if len(data[i][1][j])>k:
            rects= getRect(data[i][2][j][k],data[i][1][j][k])
            polygon1= unary_union(rects)
            polygon= unary_union([polygon,polygon1])
            val= abs(polygon.area/19.7192)
            if k!=start*10+1:
                if val<prev:
                    arr.append(prev)
                else:
                    arr.append(val)
                    prev=val
            else:
                prev=val
                arr.append(val)

    for q in range(1, len(arr)):
        if arr[q]<arr[q-1]:
            arr[q]= arr[q-1]
        if start+length >20:
            if arr[q]>arr[q-1]+0.05:
                arr[q]= arr[q-1]

    return (arr[-1])

def maxAngles(i, j, start, length):
    pmax=0
    ymax=0
 
    pin= data[i][1][j][start*10]
    yin= data[i][2][j][start*10]

    for k in range(start*10+1,start*10+length*10):
        if abs(data[i][1][j][k]-pin)>pmax:
            pmax= abs(data[i][1][j][k]-pin)
        val= abs(data[i][2][j][k]-yin)
        if val>3.14:
            val=6.28-val
        if val>ymax:
            ymax= val
            
    return(pmax,ymax)
    
def speeds(i,j, start, length):
    #i th video
    #j th user
    a=[0]*10
    pitchSpeeds= []
    yawSpeeds=[]

    
    if len(data[i][1][j])> start*10+length*10:
        pitchSpeeds= np.multiply((np.asarray(data[i][1][j][start*10+1:start*10+length*10]) - np.asarray(data[i][1][j][start*10:start*10+length*10-1])),10)
        
    if len(data[i][2][j])> start*10+length*10:
        yawSpeeds= np.multiply((np.asarray(data[i][2][j][start*10+1:start*10+length*10]) - np.asarray(data[i][2][j][start*10:start*10+length*10-1])),10)
        for k in range(len(yawSpeeds)):
            if yawSpeeds[k]>32:
                yawSpeeds[k]=62.8- yawSpeeds[k]
    pmean= np.mean(pitchSpeeds)
    ymean= np.mean(yawSpeeds)
    p25= np.percentile(pitchSpeeds,25)
    y25= np.percentile(yawSpeeds,25)
    p75= np.percentile(pitchSpeeds,75)
    y75= np.percentile(yawSpeeds,75)
    ystd=np.std(yawSpeeds)
    pstd= np.std(pitchSpeeds)

    return (y25, ymean, y75, p25,pmean, p75)


def getAllFeatures(length=2):
    features= []

    starts=[0,2,4,6,8,10,12,14,16,18,20,22,24,26]

    #start=30   ##### or 26
    #length=2

    for start in starts:
        path= folder_path+ 'extracted_features/'+str(start)+'-'+str(length+start)+'s/'
        for i in range(88):
            f= open(path+str(i)+'.txt',"r")

            arr= f.readlines()
            for j in arr:
                a= list(map(float,j.split(' ')))
                a.insert(2,start)
                features.append(a)
            f.close()

        #features2.append(arr)
    features= np.array(features)
    
    return features


###############saving all features
def saveAllFeatures(features):
    path= folder_path+ 'extracted_features/allFeatures.txt'
    f= open(path,"w")
    for i in range(len(features)):
        
        str1 = ' '.join(str(e) for e in features[i])
        f.write(str1+"\n" )

    f.close()

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="the path to the top folder")
parser.add_argument("--video", help="extract features for chunks of a single video")
args = parser.parse_args()
if args.path:
    folder_path= args.path
else:
    folder_path= setPath.setFolderPath() 

if args.video == 'all':
    a= args.video
    print('All videos in the dataset')

    data= importAggregatedDataset()

    starts=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58] 
    length=2
    print('Extracting features from viewport chunks...')
    for w in tqdm(range(len(starts))):
        start=starts[w]
        for i in range(0,len(data)):
            path= folder_path+ 'extracted_features/'+str(start)+'-'+str(length+start)+'s/'
            f= open(path+str(i)+'.txt',"w")
            
            for j in range(len(data[i][1])):
                if len(data[i][1][j])>(start+length)*10:
                    features= [i,j]
                    features.extend(yawPitchStat(i,j,start,length))
                    features.append(percentageSphereVideo(i,j,start,length))
                    features.extend(maxAngles(i,j,start,length))
                    features.extend(speeds(i,j,start,length))
                    str1 = ' '.join(str(e) for e in features)
                    f.write(str1+"\n" )
            f.close()

    features= getAllFeatures()
    print('Saving all features...')
    saveAllFeatures(features)
    print('Done')

else:
    video= args.video
    print('video number ',video)
        
    data= importAggregatedDataset()

    starts=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58] 
    length=2
    print('Extracting features from viewport chunks...')
    for w in tqdm(range(len(starts))):
        start=starts[w]
        for i in range(int(video),int(video)+1):
            path= folder_path+ 'extracted_features/'+str(start)+'-'+str(length+start)+'s/'
            f= open(path+str(i)+'.txt',"w")
            
            for j in range(len(data[i][1])):
                if len(data[i][1][j])>(start+length)*10:
                    features= [i,j]
                    features.extend(yawPitchStat(i,j,start,length))
                    features.append(percentageSphereVideo(i,j,start,length))
                    features.extend(maxAngles(i,j,start,length))
                    features.extend(speeds(i,j,start,length))
                    str1 = ' '.join(str(e) for e in features)
                    f.write(str1+"\n" )
            f.close()



