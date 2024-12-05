function [DeltaBand] = compute_delta_buzsakiMethod(lfp,timestamps,samplingrate,matfilename)
% Function to compute the broadband slow wave or delta band using Buzsaki Method. It
% computes delta band using the power spectrum slope
% lfp --> lfp data 
% timestamps --> time vector of given lfp data
% samplingrate --> Sampling rate of the data
% matfilename --> name of the mat file to save the feature

window = 2;
noverlap = window-1; %1s dt
smoothfact = 15;

lfpstruc.data = lfp;
lfpstruc.timestamps = timestamps;
lfpstruc.samplingRate = samplingrate;

%Calculate PSS
[specslope,spec] = bz_PowerSpectrumSlope(lfpstruc,window,window-noverlap,'frange',[4 90],'IRASA',true);
broadbandSlowWave = -specslope.data; %So NREM is higher as opposed to lower
t_clus = specslope.timestamps;
swFFTfreqs = specslope.freqs';
specdt = 1./specslope.samplingRate;
swFFTspec = 10.^spec.amp'; %To reverse log10 in bz_PowerSpectrumSlope

IRASAsmooth = spec.IRASAsmooth';
IRASAintercept = specslope.intercept;
IRASAslope = specslope.data;

% Remove transients before calculating SW histogram
zFFTspec = NormToInt(log10(swFFTspec)','modZ');
totz = NormToInt(abs(sum(zFFTspec,2)),'modZ');
badtimes = find(totz>3);

%Smooth and 0-1 normalize
broadbandSlowWave(badtimes) = nan;
broadbandSlowWave = smooth(broadbandSlowWave,smoothfact./specdt);

% Remove ignoretimes (after smoothing), before normalizing
if ~isempty(badtimes)
	ignoretimeIDX = isnan(broadbandSlowWave);
    broadbandSlowWave(ignoretimeIDX) = [];
    t_clus(ignoretimeIDX) = [];
end

broadbandSlowWave = bz_NormToRange(broadbandSlowWave,[0 1]);

DeltaBand.data = broadbandSlowWave;
DeltaBand.spec = swFFTspec;
DeltaBand.timestamps = t_clus;
DeltaBand.freqs = swFFTfreqs;

save(matfilename,'DeltaBand')