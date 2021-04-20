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

The extracted features will be saved in the directory /extracted_features/


*   The 





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
