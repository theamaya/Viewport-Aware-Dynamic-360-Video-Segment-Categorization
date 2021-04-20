# Viewport-Aware-Dynamic-360-Video-Segment-Categorization
This repository holds the code to reproduce the analysis and the proposed algorithms of the accepted paper #106 Viewport-aware Dynamic 360 Video Segment Categorization for NOSSDAV 2021. 

##Requirements
The following packages are required

*   numpy (ver. 1.18.1)
*   List item

##Datasets
The analysis is done for an aggregaed dataset comprising of six different public datasets of head-movement logs in viewing 360 videos. [[1]](#1) , [[2]](#2), [[3]](#3), [[4]](#4), [[5]](#5), [[6]](#6).

The total number of videos in the aggregated dataset is 88. The preprocessed dataset is in the folder 'aggregated_dataset'. 
1. all datasets are resampled at 10Hz. t=[0,0.1,0.2... ]
2. Video numbers were taken to name the files. The information about the videos are given in 'VideoMetadata'
3. Every file has 2n+1 lines where n= number of users who watched the video. First line contains the time points of sampling (in seconds).  
4. 2i-1 th line has pitch angles for the ith user, 2i th line has the yaw angles for the ith user

##Feature extraction
To implement the proposed feature extraction in Section 4.1, run the script ./feature_extraction.py with the following arguments


1.   --path ~/Viewport-Aware-Dynamic-360-Video-Segment-Categorization/ (Pass the complete path to the directory of the cloned repository)
2.   --video video_number (specify a number 0-87 to extract features for a single video. If not set, the script extracts features for all 88 videos in the dataset)

The extracted features will be saved in the directory ./extracted_features/

Eg: The path to the extracted features of viewport chunks from 10-12 seconds of the 5th video is at ./extracted_features/10-12s/5.txt . Where the nth line corresponds to the nth user. 
18 space seperated values corresponds to the following features in order


1.   List item
2.   List item


##Viewport clustering for the aggregated dataset
To implement the proposed analysis for the optimum nnumber of clusters for the aggregated datset, run the script ./optimum_VP_cluster_number_analysis.py with the following arguments

1.   --path ~/Viewport-Aware-Dynamic-360-Video-Segment-Categorization/ (Pass the complete path to the directory of the cloned repository)

The resulting analysis (Figure 5.a) will be saved to ./optimum_VP_cluster_number_analysis_results . According to the explanation given in the paper, the optimum number of clusters is selected as 10.

To implement the proposed viewport clustering for the aggregated dataset, run the script ./viewport_clustering_for_dataset.py with the following arguments

1.   --path ~/Viewport-Aware-Dynamic-360-Video-Segment-Categorization/ (Pass the complete path to the directory of the cloned repository)
2.   --nclusters (number of viewport clusters. default is 10)
3.   --saveClusters (set this to save the elemets of resulting clusters to text files)  
4.   --savePlots (set this to save the plots in figure 7)  
4.   --eval (set this to run the evaluation explained in Section 4.2 and save the figure 6) 

##Comparing Viewport clustering algorithms 

To implement the comparison of the proposed algorithm with the results of Spherical-clustering ........ and reproduce the results of Figure 3, run the script ./comparing_vp_clustering_algorithm.py with the following arguments
1.   --path ~/Viewport-Aware-Dynamic-360-Video-Segment-Categorization/ (Pass the complete path to the directory of the cloned repository).

The results will be saved to ./comparing_vp_clustering_algorithm_results/

##Dynamic Video Segement Categorization for the aggregated dataset
To implement the proposed analysis for the optimum nnumber of video categories for the aggregated datset, run the script ./optimum_video_categories_number_analysis.py with the following arguments

1.   --path ~/Viewport-Aware-Dynamic-360-Video-Segment-Categorization/ (Pass the complete path to the directory of the cloned repository)
2.   ---viewportClusters (number of viewport clusters. default is 10)

The resulting analysis (Figure 10.a) will be saved to ./optimum_video_categories_number_analysis_results . According to the explanation given in the paper, the optimum number of video categories corresponding to 10 viewport clusters is selected as 6.

To implement the proposed video categorization for the aggregated dataset, run the script ./video_categorization_all.py with the following arguments

1.   --path ~/Viewport-Aware-Dynamic-360-Video-Segment-Categorization/ (Pass the complete path to the directory of the cloned repository)
2.   --viewportClusters (number of viewport clusters. default is 10)
3.   --videoCats (number of video categories. default is 6)
4.   --saveClusters (set this to save the elemets of resulting clusters to text files)  
5.   --savePlots (set this to save the plots in figure 9)  
6.   --eval (set this to run the evaluation explained in Section 4.2 and save the figure 10.b) 

## References
<a id="1">[1]</a> 
Xavier Corbillon, Francesca De Simone, and Gwendal Simon. 2017. 360-Degree Video Head Movement Dataset. In Proceedings of the 8th ACM on Multimedia Systems Conference. ACM, Taipei Taiwan, 199–204. https://doi.org/10.1145/3083187.3083215 

<a id="2">[2]</a> 
Wen-ChihLo, Ching-LingFan, Jean Lee, Chun-Ying Huang, Kuan-Ta Chen,and Cheng-Hsin Hsu.2017. 360° Video Viewing Dataset in Head-Mounted Virtual Reality.In Proceedings of the 8th ACM on Multimedia Systems Conference.ACM, Taipei Taiwan, 211–216. https://doi.org/10.1145/3083187.3083219 

<a id="3">[3]</a> 
Yanan Bao, Huasen Wu, Tianxiao Zhang, Albara Ah Ramli, and Xin Liu.2016. Shooting a moving target: Motion-prediction-based transmission for 360-degree videos. In 2016 IEEE International Conference on Big Data (Big Data). IEEE,Washington DC, USA ,1161–1170. https://doi.org/10.1109/BigData.2016.7840720 

<a id="4">[4]</a> 
Chenglei Wu, Zhihao Tan, Zhi Wang, and Shiqiang Yang. 2017. A Dataset for Exploring User Behaviors in VR Spherical Video Streaming. In Proceedings of the 8th ACM on Multimedia Systems Conference. ACM, Taipei Taiwan, 193–198. https://doi.org/10.1145/3083187.3083210 

<a id="5">[5]</a> 
Yu Guan, Chengyuan Zheng, Xinggong Zhang, Zongming Guo, and Junchen Jiang.2019. Pano: optimizing 360° video streaming with a better understanding of quality perception. In Proceedings of the ACM Special Interest Group on Data Communication. ACM, Beijing China,394–407. https://doi.org/10.1145/3341302.3342063 

<a id="6">[6]</a> 
Afshin Taghavi Nasrabadi, Aliehsan Samiei, Anahita Mahzari, Ryan P. McMahan, Ravi Prakash, Mylène C. Q. Farias, and Marcelo M. Carvalho. 2019. A taxonomy
and dataset for 360° videos. In Proceedings of the 10th ACM Multimedia Systems Conference. ACM, Amherst Massachusetts, 273–278. https://doi.org/10.1145/3304109.3325812 

###add references of the 3 VP clustering algorithms

