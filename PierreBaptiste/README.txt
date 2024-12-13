To use the thresholding method, you must use the GetData code. m and getmetrics_V2.m
First, you need to download the toolbox file that contains all the codes that allow analysis.
You must then use the GetData. m code to retrieve the three signals needed for analysis.
The code is segmented into different parrties, one part per signal.
Once the signals are recovered, it is possible to run the code getmetrcis_V2.m in order to have the classification of the sleep phases.
To finish the code Correlation_bz_manual. m is a code that allows to make the comparison between the file created by the code and the file classified manually.

The Acceleration_Data_Curation_version5.m code loads raw accelerometer data, reduces it by sampling to reduce noise, and normalizes signals for quantifying motion.