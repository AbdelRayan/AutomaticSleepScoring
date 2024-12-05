function [ThetaBand] = compute_theta_buzsakiMethod(lfp,timestamps,samplingrate,matfilename)
% Function to compute the narrowband theta ratio using Buzsaki Method. It
% computes theta band using the power spectrum slope
% lfp --> lfp data 
% timestamps --> time vector of given lfp data
% samplingrate --> Sampling rate of the data
% matfilename --> name of the mat file to save the feature

f_all = [2 20];
f_theta = [5 10];
window = 2;
noverlap = window - 1;
smoothfact = 15;

lfpstruc.data = lfp;
lfpstruc.timestamps = timestamps;
lfpstruc.samplingRate = samplingrate;
%Calculate PSS
[specslope,spec] = bz_PowerSpectrumSlope(lfpstruc,window,window-noverlap,...
    'nfreqs',200,'frange',f_all,'IRASA',true);
t_thclu = specslope.timestamps;
specdt = 1./specslope.samplingRate;
thFFTspec = specslope.resid';
thFFTspec(thFFTspec<0)=0;

IRASAsmooth_th = spec.IRASAsmooth';
thFFTspec_raw = 10.^spec.amp';

% Remove transients before calculating SW histogram
zFFTspec = NormToInt(spec.amp,'modZ');
totz = NormToInt(abs(sum(zFFTspec,2)),'modZ');
badtimes_TH = find(totz>3);

thFFTfreqs = specslope.freqs';
thfreqs = (thFFTfreqs>=f_theta(1) & thFFTfreqs<=f_theta(2));
thratio = max((thFFTspec(thfreqs,:)),[],1);

thratio(badtimes_TH) = nan;     %Remove transients
thratio = smooth(thratio,smoothfact./specdt);

% Remove ignoretimes (after smoothing), before normalizoing
if ~isempty(badtimes_TH) 
	ignoretimeIDX = isnan(thratio);
    thratio(ignoretimeIDX) = [];
    t_thclu(ignoretimeIDX) = [];
end

thratio = bz_NormToRange(thratio,[0 1]);

ThetaBand.data = thratio;
ThetaBand.spec = thFFTspec;
ThetaBand.timestamps = t_thclu;
ThetaBand.freqs = thFFTfreqs;

save(matfilename,'ThetaBand');

