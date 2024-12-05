% The aim of this script is to do preprocessing of accelerometer data
% accroding to the method cited by Buzsaki 
clear all; close all; clc

% 1. Upload the raw data 
[file,path]=uigetfile('*.cont*','Select the accelerometer file(s) (select multiple if possible)', 'MultiSelect', 'on');
patIDacc = {};
for jj=1:length(file)
    tmp = file(jj);
    patIDacc{jj} = strcat([path,tmp{1}]);
end

% 2. reading the data from the openephys 
ACC_RES = [];
disp('Loading and downsampling of the accelerometer data') 
for jj=1:length(patIDacc)
    [Dataacc, TimeVect, ~] = load_open_ephys_data(patIDacc{jj});
%     Dataacc=filtfilt(b,a,Dataacc);
%     Dataacc=downsample(Dataacc,acq_fhz/500);
    ACC_RES = [ACC_RES  Dataacc]; 
end
disp('Finished reading the accelerometer data')
%% 3. downsample the data to reduce noise and flickering of the baseline 
% the data is downsampled to 100 Hz --> could be reduced more 
NDown = 10000; % how many times you want to reduce the sampling of the data 
FsDown = 20000/NDown;
g_x = decimate(ACC_RES(:,1),NDown,'FIR');
g_x_mean = mean(g_x);
g_x = g_x - g_x_mean;
g_y = decimate(ACC_RES(:,2),NDown,'FIR');
g_y_mean = mean(g_y);
g_y = g_y - g_y_mean;
g_z = decimate(ACC_RES(:,3),NDown,'FIR');
g_z_mean = mean(g_z);
g_z = g_z - g_z_mean;
%%
MotionMatrix = [g_x g_y g_z]';
meeg = abs(zscore(MotionMatrix')');
meeg = sum(meeg, 1);
% meeg = locdetrend(meeg,FsDown,[.1 .01]); 
forder = 500;
forder = ceil(forder/2)*2;
EEGSR = FsDown;
lowband = 0.1;
highband = 1;
% firfiltb = fir1(forder,[lowband/EEGSR*2,highband/EEGSR*2]);
% meegF = filter2(firfiltb,  meeg);
TimeVectDown = linspace(0,numel(meeg)/FsDown,numel(meeg));
motion = mean(reshape(meeg(1:(length(meeg) - mod(length(meeg), FsDown))), FsDown, []), 1);
hilbertAccelero = hilbert(meeg);    
AcceleroAmp = abs(hilbertAccelero);
% motion2 = locdetrend(AcceleroAmp,FsDown,[.1 .01]); 
motion3 = bz_NormToRange(AcceleroAmp,[0 1]);
%% plotting the outcome 
figure 
plot(TimeVectDown,motion3,'LineWidth',2)
hold on 
% plot(TimeVectDown,meegF,'r','LineWidth',2)
xlabel('Time [s]')
ylabel('g-level value')
box off 
set(gca,'FontSize',15,'LineWidth',1.5,'FontWeight','bold','FontName','Times')
set(gcf,'Color','w')
% Sauvegarde des données importantes dans un fichier .mat
save('ProcessedAccelerometerData.mat', 'TimeVectDown', 'meeg', 'motion3');

% export_fig('AccelerometerDataExample','-pdf','-r300','-q70')
% %% another plot for comaprison
% figure 
% % plot(EMGFromLFP.timestamps,EMGFromLFP.data,'LineWidth',2)
% hold on 
% plot(TimeVectDown,meegFNorm,'r','LineWidth',2)
% xlabel('Time [s]')
% ylabel('g-level value')
% box off 
% set(gca,'FontSize',15,'LineWidth',1.5,'FontWeight','bold','FontName','Times')
% set(gcf,'Color','w')
% %% Handling the EMG data like Buzsaki 
% smoothfact1 = 15;
% smoothfact2 = 1250;
% dtEMG = 1/EMGFromLFP.samplingFrequency;
% EMGFromLFP.smoothed = smooth(EMGFromLFP.data,smoothfact1/dtEMG,'moving');
% AcceleroMeterSmooth = smooth(AcceleroAmp,smoothfact2/dtEMG,'moving');
% meegFNorm = bz_NormToRange(AcceleroMeterSmooth,[0 1]);