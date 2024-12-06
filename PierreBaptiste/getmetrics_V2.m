% Load the data
data = load('/home/genzellab/tresholding/CBD/6/Rat_OS_Ephys_cbd_chronic_Rat6_411357_SD14_OR_20210729/2021-07-29_16-08-42_posttrial5/DeltaBandData.mat');  % Replace 'SWLFP.mat' with your file name
broadbandSlowWave = data.DeltaBand.data;  % Ensure the variable in the file is correct

% If the file also contains timestamps, load them:
if isfield(data, 'DeltaBand') && isfield(data.DeltaBand, 'timestamps')
    t_clus = data.DeltaBand.timestamps;  % Retrieve associated timestamps
else
    % If timestamps are not provided, generate them based on sampling frequency
    sf_LFP = 2500;  % Example sampling frequency (in Hz, adjust according to your data)
    t_clus = (0:length(broadbandSlowWave)-1) / sf_LFP;
end

% Initialization of SWweights and other necessary parameters for PSS
SWweights = 'PSS';  % Method choice for Power Spectrum Slope
IRASA = false;      % Set IRASA according to your preference

% Parameter for the down-sampling factor
downsamplefactor = 2;  % Adjust this factor based on your needs

% Down-sampling LFP data
% swLFP = downsample(data.DeltaBand.data, downsamplefactor);
swLFP = data.DeltaBand.data;

% Load EMG data
EMG = load('/home/genzellab/tresholding/CBD/6/Rat_OS_Ephys_cbd_chronic_Rat6_411357_SD14_OR_20210729/2021-07-29_16-08-42_posttrial5/ProcessedData.mat');
motiondata = EMG.ProcessData.motion3;

onSticky = false;

if onSticky
    stickySW = true; stickyTH = false; stickyMotion = true;
else
    stickySW = false; stickyTH = false; stickyMotion = false;
end

% Calculate per-bin weights onto SlowWave
% assert(isequal(swFFTfreqs, SWfreqlist), ...
%       'Spectrogram freqs. are not as expected...');
% broadbandSlowWave = zFFTspec * SWweights';

% Initial parameters for the histogram and threshold search
numpeaks = 1;
numbins = 12;

% Loop to find a bimodal histogram and peaks
while numpeaks ~= 2
    % Use histogram if hist raises issues
    [swhist, swhistbins] = histcounts(broadbandSlowWave, numbins);

    % Adjust bin centers for compatibility with findpeaks
    swhistbins = (swhistbins(1:end-1) + swhistbins(2:end)) / 2;

    [PKS, LOCS] = findpeaks(swhist, 'NPeaks', 2, 'SortStr', 'descend');
    LOCS = sort(LOCS);
    numbins = numbins + 1;
    numpeaks = length(LOCS);
end

% Calculate Hartigan's test for bimodality
SWdiptest = bz_hartigansdiptest(broadbandSlowWave);

% Find the threshold between the two peaks
betweenpeaks = swhistbins(LOCS(1):LOCS(2));
[dip, diploc] = findpeaks(-swhist(LOCS(1):LOCS(2)), 'NPeaks', 1, 'SortStr', 'descend');
swthresh = betweenpeaks(diploc);

% Define time points corresponding to SWS (NREM) states
NREMtimes = (broadbandSlowWave > swthresh);

% Then Divide Motion
numpeaks = 1;
numbins = 12;
if sum(isnan(motiondata)) > 0
    error('Motion data seems to contain NaN values...');
end

while numpeaks ~= 2
    [MotionHist, MotionHistBins] = hist(motiondata, numbins);

    [PKS, LOCS] = findpeaks_SleepScore([0 MotionHist], 'NPeaks', 2);

    LOCS = sort(LOCS) - 1;
    numbins = numbins + 1;
    numpeaks = length(LOCS);

    if numpeaks == 100
        display('Something is wrong with your Motion Signal');
        return;
    end
end

Motiondiptest = bz_hartigansdiptest(motiondata);
betweenpeaks = MotionHistBins(LOCS(1):LOCS(2));
[dip, diploc] = findpeaks_SleepScore(-MotionHist(LOCS(1):LOCS(2)), 'NPeaks', 1, 'SortStr', 'descend');

MotionThresh = betweenpeaks(diploc);

MOVtimes = (broadbandSlowWave(:) < swthresh & motiondata(:) > MotionThresh);


%% Then Divide Theta (During NonMoving)

% Assume t_LFP is the time vector
% Also assume sf_LFP is the sampling frequency (2500 Hz)
window = 2;       % Window of 5000 samples (total signal duration)
noverlap = 1;     % No overlap
sf_LFP = 2500;       % Sampling frequency (in Hz)
Theta = load('/home/genzellab/tresholding/CBD/6/Rat_OS_Ephys_cbd_chronic_Rat6_411357_SD14_OR_20210729/2021-07-29_16-08-42_posttrial5/ThetaBandData.mat');
thratio = Theta.ThetaBand.data;  % Load theta band data
t_LFP = Theta.ThetaBand.timestamps;  % Load time stamps
smoothfact = 15;     % Smoothing factor
%ThIRASA = false;     % IRASA flag (set to false);

%%
numpeaks = 1;
numbins = 12;
while numpeaks ~=2 && numbins <=25
    %[THhist,THhistbins]= hist(thratio(SWStimes==0 & MOVtimes==0),numbins);
    if length(thratio) < length(MOVtimes)
        thratio(end+1:length(MOVtimes)) = thratio(end);
    end


    [THhist,THhistbins]= hist(thratio(MOVtimes==0),numbins);

    [PKS,LOCS] = findpeaks_SleepScore(THhist,'NPeaks',2,'SortStr','descend');
    LOCS = sort(LOCS);
    numbins = numbins+1;
    numpeaks = length(LOCS);
end

%THdiptest = bz_hartigansdiptest(thratio(MOVtimes==0));

if numpeaks ~= 2
	display('No bimodal dip found in theta. Trying to exclude NREM...')

    numbins = 12;
    %numbins = 15; %for Poster...
    while numpeaks ~=2 && numbins <=25

        [THhist,THhistbins]= hist(thratio(NREMtimes==0 & MOVtimes==0),numbins);

        [PKS,LOCS] = findpeaks_SleepScore(THhist,'NPeaks',2,'SortStr','descend');
        
        LOCS = sort(LOCS);
        numbins = numbins+1;
        numpeaks = length(LOCS);

    end
    
	try
        THdiptest = bz_hartigansdiptest(thratio(NREMtimes==0 & MOVtimes==0));
    catch
	end
end

if length(PKS)==2
    betweenpeaks = THhistbins(LOCS(1):LOCS(2));
    [dip,diploc] = findpeaks_SleepScore(-THhist(LOCS(1):LOCS(2)),'NPeaks',1,'SortStr','descend');

    THthresh = betweenpeaks(diploc);

    REMtimes = (broadbandSlowWave<swthresh & motiondata<MotionThresh & thratio>THthresh);
else
    display('No bimodal dip found in theta. Use TheStateEditor to manually select your threshold (hotkey: A)')
    THthresh = 0;
%     REMtimes =(broadbandSlowWave<swthresh & EMG<EMGthresh);
end

histsandthreshs = v2struct(swhist,swhistbins,swthresh,MotionHist,MotionHistBins,...
    MotionThresh,THhist,THhistbins,THthresh,...
    stickySW,stickyTH,stickyMotion);

%% Ouput Structure: StateScoreMetrics
% LFPparams = SleepScoreLFP.params;
% WindowParams.window = window;
% WindowParams.smoothwin = smoothfact;
% THchanID = SleepScoreLFP.THchanID; SWchanID = SleepScoreLFP.SWchanID;

SleepScoreMetrics = v2struct(broadbandSlowWave,thratio,motiondata,...
    t_clus,histsandthreshs,...
    Motiondiptest,SWdiptest);
save('/home/genzellab/tresholding/CBD/6/Rat_OS_Ephys_cbd_chronic_Rat6_411357_SD14_OR_20210729/2021-07-29_16-08-42_posttrial5/results_ClusterStates_GetMetrics.mat','SleepScoreMetrics');

%StatePlotMaterials = v2struct(t_clus,swFFTfreqs,swFFTspec,thFFTfreqs,thFFTspec);


%% Determine States
[ints, idx, MinTimeWindowParms] = ClusterStates_DetermineStates(SleepScoreMetrics, [], histsandthreshs, MOVtimes, NREMtimes);
save('/home/genzellab/tresholding/CBD/6/Rat_OS_Ephys_cbd_chronic_Rat6_411357_SD14_OR_20210729/2021-07-29_16-08-42_posttrial5/resultats_ClusterStates_DetermineStates.mat', 'ints', 'idx', 'MinTimeWindowParms');
disp('test')
IDX = interp1(idx.timestamps, idx.states,t_clus,'nearest');

%% Figure: Clustering

colormat = [[0 0 0];[nan nan nan];[0 0 1];[nan nan nan];[1 0 0];[nan nan nan]];
if any(IDX==0) || any(isnan(IDX)) 
    IDX(IDX==0 | isnan(IDX)) = 6;
end
coloridx = colormat(IDX,:);



    figure

        
    subplot(1,3,[2,3])
    hold all
    
    scatter3(broadbandSlowWave(IDX~=5), thratio(IDX~=5), motiondata(IDX~=5), ...
             2, coloridx(IDX~=5, :), 'filled') 
    
    scatter3(broadbandSlowWave(IDX==5), thratio(IDX==5), motiondata(IDX==5), ...
             20, 'r', 'x') 
    
    view(133.7, 18.8);
    grid on
    xlabel('Broadband SW'); ylabel('Narrowband Theta'); zlabel('EMG')

	subplot(3,3,1)
        hold on
        bar(swhistbins,swhist,'FaceColor','none','barwidth',0.9,'linewidth',2)
        plot([swthresh swthresh],[0 max(swhist)],'r')
        xlabel('Slow Wave')
        title('Step 1: Broadband for NREM')
	subplot(3,3,4)
        hold on
        bar(MotionHistBins,MotionHist,'FaceColor','none','barwidth',0.9,'linewidth',2)
        plot([MotionThresh MotionThresh],[0 max(MotionHist)],'r')
        xlabel('EMG')
        title('Step 2: EMG for Muscle Tone')
	subplot(3,3,7)
        hold on
        bar(THhistbins,THhist,'FaceColor','none','barwidth',0.9,'linewidth',2)
        plot([THthresh THthresh],[0 max(THhist)],'r')
        xlabel('Theta')
        title('Step 3: Theta for REM')
        
    %sgtitle('Presleep 2021-07-20 10-27-40')% 	saveas(gcf,[figloc,recordingname,'_SSCluster3D'],'jpeg')