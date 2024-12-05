function [ints, idx, MinTimeWindowParms] = ClusterStates_DetermineStates(...
    SleepScoreMetrics,MinTimeWindowParms,histsandthreshs,MOVtimes, NREMtimes)

% Initialisation des paramètres minimums (si nécessaires)
if exist('MinTimeWindowParms','var') && ~isempty(MinTimeWindowParms)
     v2struct(MinTimeWindowParms)
else
    minSWSsecs = 6;
    minWnexttoREMsecs = 6;
    minWinREMsecs = 6;       
    minREMinWsecs = 6;
    minREMsecs = 6;
    minWAKEsecs = 6;
    MinTimeWindowParms = v2struct(minSWSsecs,minWnexttoREMsecs,minWinREMsecs,...
        minREMinWsecs,minREMsecs,minWAKEsecs);
end

% Traitement des variables de seuils
if exist('histsandthreshs','var')
    hat2 = histsandthreshs;
    v2struct(SleepScoreMetrics)
    histsandthreshs = hat2;
else
    v2struct(SleepScoreMetrics)
end
v2struct(histsandthreshs)

% % Bimodalité de theta
%     numpeaks = 1;
%     numbins = 12;
%     while numpeaks ~= 2 && numbins <= 25
%         [THhist, THhistbins] = hist(thratio(MOVtimes == 0), numbins);
%         [PKS, LOCS] = findpeaks_SleepScore(THhist, 'NPeaks', 2, 'SortStr', 'descend');
%         LOCS = sort(LOCS);
%         numbins = numbins + 1;
%         numpeaks = length(LOCS);
%     end
% 
%     %THdiptest = bz_hartigansdiptest(thratio(MOVtimes == 0));
% 
%     if numpeaks ~= 2
%         display('No bimodal dip found in theta. Trying to exclude NREM...')
%         numbins = 12;
%         while numpeaks ~= 2 && numbins <= 25
%             [THhist, THhistbins] = hist(thratio(NREMtimes == 0 & MOVtimes == 0), numbins);
%             [PKS, LOCS] = findpeaks_SleepScore(THhist, 'NPeaks', 2, 'SortStr', 'descend');
%             LOCS = sort(LOCS);
%             numbins = numbins + 1;
%             numpeaks = length(LOCS);
%         end
%         try
%             THdiptest = bz_hartigansdiptest(thratio(NREMtimes == 0 & MOVtimes == 0));
%         catch
%         end
%     end
% 
%     % Détermination du seuil THthresh si bimodalité détectée
%     if length(PKS) == 2
%         betweenpeaks = THhistbins(LOCS(1):LOCS(2));
%         [dip, diploc] = findpeaks_SleepScore(-THhist(LOCS(1):LOCS(2)), 'NPeaks', 1, 'SortStr', 'descend');
%         THthresh = betweenpeaks(diploc);
%     else
%         display('No bimodal dip found in theta. Use TheStateEditor to manually select your threshold (hotkey: A)')
%         THthresh = 0;
%     end

if ~exist('stickySW','var'); stickySW = false; end
if ~exist('stickyTH','var'); stickyTH = false; end
if ~exist('stickyMotion','var'); stickyMotion = false; end


[~,~,~,~,NREMtimes] = bz_BimodalThresh(broadbandSlowWave(:),'startbins',15,...
    'setthresh',swthresh,'diptest',false,'Schmidt',stickySW,'0Inf',true);

[~,~,~,~,hightheta] = bz_BimodalThresh(thratio(:),'startbins',15,...
    'setthresh',THthresh,'diptest',false,'Schmidt',stickyTH,'0Inf',true);

[~,~,~,~,highMotion] = bz_BimodalThresh(motiondata(:),'startbins',15,...
    'setthresh',MotionThresh,'diptest',false,'Schmidt',stickyMotion,'0Inf',true);

REMtimes = (~NREMtimes & ~highMotion & hightheta);

%ACTIVE/QUIET WAKE:
WAKEtimes = ~NREMtimes & ~REMtimes;
QWAKEtimes =  WAKEtimes & ~hightheta; %Used later if QWake scored
% Récupération des timestamps pour les états
IDX.statenames = {'WAKE','','NREM','','REM'};
IDX.timestamps = t_clus; %Timestamps pulled from clustering (in ClusterStates_GetMetrics)
IDX.states = zeros(size(IDX.timestamps));

% Attribution des états selon les intervalles
IDX.states(NREMtimes) = 3;
IDX.states(REMtimes) = 5;
IDX.states(WAKEtimes) = 1;

% Si score QWAKE activé
if ~exist('scoreQW','var'); scoreQW = false; end
if scoreQW
    IDX.states(QWAKEtimes) = 2;
    IDX.statenames{2} = 'QWAKE';
end

% Conversion en intervals
% disp('IDX :')
% disp(IDX)
INT = bz_IDXtoINT(IDX);
% disp('INT :')
% disp(INT); % Affiche la structure de INT
IDX = bz_INTtoIDX(INT,'statenames',{'WAKE','','NREM','','REM'});
%disp(INT.NREMstate);

% Diagnostic pour vérifier la taille et le contenu de NREMstate et shortSints
% disp('Taille de INT.NREMstate :');
% disp(size(INT.NREMstate));
% disp('Contenu de INT.NREMstate :');
% disp(INT.NREMstate);

% Calcul de la durée des intervalles NREM
Sdur = diff(INT.NREMstate, [], 2);
shortSints = Sdur <= minSWSsecs;

% Vérification de shortSints
% disp('Taille de shortSints :');
% disp(size(shortSints));
% disp('Contenu de shortSints :');
% disp(shortSints);

% Conversion de shortSints en indices explicites
shortSintsIdx = find(shortSints);

% Vérification de la forme de INT.NREMstate(shortSintsIdx, :)
% disp('Taille de INT.NREMstate(shortSintsIdx, :) :');
% disp(size(INT.NREMstate(shortSintsIdx, :)));
% disp('Contenu de INT.NREMstate(shortSintsIdx, :) :');
% disp(INT.NREMstate(shortSintsIdx, :));

INT.WAKEstate= double(INT.WAKEstate);
INT.NREMstate = double(INT.NREMstate);
INT.REMstate = double(INT.REMstate);


% Étape 1: Créer une variable d'intervalles avant d'appeler InIntervals
intervals = INT.NREMstate(shortSintsIdx, :);
% Vérification de la structure des intervalles
% disp('Intervals:');
% disp(intervals);

shortSidx = InIntervals(IDX.timestamps,intervals);
IDX.states(shortSidx) = 1;   
INT = bz_IDXtoINT(IDX);


%Short WAKE (next to REM) -> REM
Wdur = diff(INT.WAKEstate,[],2);
shortWints = Wdur<=minWnexttoREMsecs;
[~,~,WRints,~] = FindIntsNextToInts(INT.WAKEstate,INT.REMstate,1);
[~,~,~,RWints] = FindIntsNextToInts(INT.REMstate,INT.WAKEstate,1);
shortWRidx = InIntervals(IDX.timestamps,INT.WAKEstate(shortWints & (WRints | RWints),:));
IDX.states(shortWRidx) = 5;   
INT = bz_IDXtoINT(IDX);

%%
%Short REM (in WAKE) -> WAKE
Rdur = diff(INT.REMstate,[],2);
shortRints =Rdur<=minREMinWsecs;
[~,~,~,WRints] = FindIntsNextToInts(INT.WAKEstate,INT.REMstate,1);
[~,~,RWints,~] = FindIntsNextToInts(INT.REMstate,INT.WAKEstate,1);
shortRWidx = InIntervals(IDX.timestamps,INT.REMstate(shortRints & (WRints & RWints),:));
IDX.states(shortRWidx) = 1;   
INT = bz_IDXtoINT(IDX);

%%
%Remaining Short REM (in NREM) -> WAKE
Rdur = diff(INT.REMstate,[],2);
shortRints = Rdur<=minREMsecs;
shortRidx = InIntervals(IDX.timestamps,INT.REMstate(shortRints,:));
IDX.states(shortRidx) = 1;   
INT = bz_IDXtoINT(IDX);

%WAKE   (to SWS)     essentiall a minimum MA time
Wdur = diff(INT.WAKEstate,[],2);
shortWints = Wdur<=minWAKEsecs;
shortWidx = InIntervals(IDX.timestamps,INT.WAKEstate(shortWints,:));
IDX.states(shortWidx) = 3;   
INT = bz_IDXtoINT(IDX);

%SWS  (to NonMOV)
Sdur = diff(INT.NREMstate,[],2);
shortSints = Sdur<=minSWSsecs;
shortSidx = InIntervals(IDX.timestamps,INT.NREMstate(shortSints,:));
IDX.states(shortSidx) = 1;   
INT = bz_IDXtoINT(IDX);


% Sortie finale
ints = INT;
idx = IDX;

end
