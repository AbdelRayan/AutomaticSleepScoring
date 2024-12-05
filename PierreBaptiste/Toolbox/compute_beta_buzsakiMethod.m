function [BetaBand] = compute_beta_buzsakiMethod(lfp,timestamps,samplingrate,matfilename)
% Function to compute the narrowband beta ratio using Buzsaki Method. It
% computes beta band using the power spectrum slope
% lfp --> lfp data 
% timestamps --> time vector of given lfp data
% samplingrate --> Sampling rate of the data
% matfilename --> name of the mat file to save the feature

f_all = [5 25];
f_beta = [9 18];
window = 2;
noverlap = window - 1;
smoothfact = 15;

lfpstruc.data = lfp;
lfpstruc.timestamps = timestamps;
lfpstruc.samplingRate = samplingrate;
%Calculate PSS
[specslope,spec] = bz_PowerSpectrumSlope(lfpstruc,window,window-noverlap,...
    'nfreqs',200,'frange',f_all,'IRASA',true);
t_bclu = specslope.timestamps;
specdt = 1./specslope.samplingRate;
betaFFTspec = specslope.resid';
betaFFTspec(betaFFTspec<0)=0;

IRASAsmooth_b = spec.IRASAsmooth';
betaFFTspec_raw = 10.^spec.amp';

% Remove transients before calculating SW histogram
zFFTspec = NormToInt(spec.amp,'modZ');
totz = NormToInt(abs(sum(zFFTspec,2)),'modZ');
badtimes_B = find(totz>3);

betaFFTfreqs = specslope.freqs';
betafreqs = (betaFFTfreqs>=f_beta(1) & betaFFTfreqs<=f_beta(2));
bratio = max((betaFFTspec(betafreqs,:)),[],1);

bratio(badtimes_B) = nan;     %Remove transients
bratio = smooth(bratio,smoothfact./specdt);

% Remove ignoretimes (after smoothing), before normalizoing
if ~isempty(badtimes_B) 
	ignoretimeIDX = isnan(bratio);
    bratio(ignoretimeIDX) = [];
    t_bclu(ignoretimeIDX) = [];
end

bratio = bz_NormToRange(bratio,[0 1]);

BetaBand.data = bratio;
BetaBand.spec = betaFFTspec;
BetaBand.timestamps = t_bclu;
BetaBand.freqs = betaFFTfreqs;

save(matfilename,'BetaBand');

