clear all;

addpath(genpath('/home/genzellab/tresholding/FMAToolbox'));
savepath;

% Load data from the .mat file
data = load('/home/genzellab/tresholding/CBD/6/Rat_OS_Ephys_cbd_chronic_Rat6_411357_SD14_OR_20210729/2021-07-29_16-08-42_posttrial5/PFC_100_CH12_0.continuous.mat');

% Assume that LFP data, timestamps, and sampling rate are stored in variables called
% 'lfp', 'timestamps', and 'samplingrate', respectively. Adjust the names based on your file.
PFC = data.PFC; % Replace 'lfp' with the correct variable name in your file
samplingrate = 2500; % Replace with the correct variable name if needed
num_samples = length(PFC); % Number of samples in the LFP signal
timestamps = (0:num_samples-1) / samplingrate; % Create a series of timestamps

% Specify the filename to save the results
matfilename = 'DeltaBandData.mat';

% Call the function to compute the delta band
DeltaBand = compute_delta_buzsakiMethod(PFC, timestamps, samplingrate, matfilename);

% Display the results (optional)
disp(DeltaBand);

%%
% Load data from the .mat file
data = load('/home/genzellab/tresholding/CBD/6/Rat_OS_Ephys_cbd_chronic_Rat6_411357_SD14_OR_20210729/2021-07-29_16-08-42_posttrial5/HPC_100_CH15_0.continuous.mat');

% Adjust the variable names based on your file
HPC = data.HPC; 
samplingrate = 2500;
num_samples = length(HPC);
timestamps = (0:num_samples-1) / samplingrate;

% Specify the filename to save the results
matfilename = 'ThetaBandData.mat';

% Call the function to compute the theta band by adjusting the frequency range
ThetaBand = compute_theta_buzsakiMethod(HPC, timestamps, samplingrate, matfilename);

% Display the results (optional)
disp(ThetaBand);

%%
% Load data from the .mat file
data = load('/home/genzellab/tresholding/CBD/6/Rat_OS_Ephys_cbd_chronic_Rat6_411357_SD14_OR_20210729/2021-07-29_16-08-42_posttrial5/PFC_100_CH12_2_0.continuous.mat');

% Adjust the variable names based on your file
PFC = data.PFC; 
samplingrate = 2500;
num_samples = length(PFC);
timestamps = (0:num_samples-1) / samplingrate;

% Specify the filename to save the results
matfilename = 'BetaBandData.mat';

% Call the function to compute the beta band by adjusting the frequency range
BetaBand = compute_beta_buzsakiMethod(PFC, timestamps, samplingrate, matfilename);

% Display the results (optional)
disp(BetaBand);

%%
% Define the sampling frequency
samplingFrequency = 2500; % Hz

% Load data from the .mat files
dataPFC = load('/home/genzellab/tresholding/CBD/4/Rat_OS_Ephys_cbd_chronic_Rat4_407699_SD4_OD_20210607/22021-06-07_14-39-27_posttrial5Rat_OS_Ephys_cbd_chronic_Rat4_407699_SD4_OD_20210607/22021-06-07_14-39-27_posttrial5/PFC_100_CH22_0.continuous.mat');
dataHPC = load('/home/genzellab/tresholding/CBD/4/Rat_OS_Ephys_cbd_chronic_Rat4_407699_SD4_OD_20210607/22021-06-07_14-39-27_posttrial5Rat_OS_Ephys_cbd_chronic_Rat4_407699_SD4_OD_20210607/22021-06-07_14-39-27_posttrial5/HPC_100_CH18_0.continuous.mat');

% Assume that LFP data for PFC and HPC are stored in variables named 'PFC' and 'HPC'.
sig1 = dataPFC.PFC; % Replace with the correct name if necessary
sig2 = dataHPC.HPC; % Replace with the correct name if necessary

% Define parameters for EMG computation
targetSampling = 1; % Example of target frequency for EMG (modifiable)
smoothWindow = 10; % Smoothing window for EMG (modifiable)
matfilename = 'EMGData.mat'; % Filename to save the results

% Compute EMG using the Buzsaki method
EMGFromLFP = compute_emg_buzsakiMethod(targetSampling, samplingFrequency, sig1, sig2, smoothWindow, matfilename);

% Display the results
disp(EMGFromLFP);

%%
% Load delta, theta, and beta bands (ensure they were computed earlier)
DeltaBand = load('/home/genzellab/tresholding/CBD/6/Rat_OS_Ephys_cbd_chronic_Rat6_411357_SD14_OR_20210729/2021-07-29_16-08-42_posttrial5/DeltaBandData.mat');
ThetaBand = load('/home/genzellab/tresholding/CBD/6/Rat_OS_Ephys_cbd_chronic_Rat6_411357_SD14_OR_20210729/2021-07-29_16-08-42_posttrial5/ThetaBandData.mat');
BetaBand = load('/home/genzellab/tresholding/CBD/6/Rat_OS_Ephys_cbd_chronic_Rat6_411357_SD14_OR_20210729/2021-07-29_16-08-42_posttrial5/BetaBandData.mat');

% Extract the required data
DeltaBand = DeltaBand.DeltaBand; % Adjust if the name is different
ThetaBand = ThetaBand.ThetaBand; % Adjust if the name is different
BetaBand = BetaBand.BetaBand; % Adjust if the name is different

% Extract the EMG signal
EMG = EMGFromLFP.smoothed; % Ensure 'smoothed' is the correct field

% Channel name
channelname = 'Presleep 2019-06-13 14-40-49'; % Change as needed
% Call the FeaturePlots function
FeaturePlots(DeltaBand, ThetaBand, BetaBand, EMG, channelname);
