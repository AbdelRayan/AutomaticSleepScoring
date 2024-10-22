from notebooks.buzsaki_method.src.utils import get_sequences, ensure_duration, get_segments
from notebooks.buzsaki_method.src.utils import _detect_troughs, _despine_axes, create_hypnogram
from detect_phasic import _compute_hilbert

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# * Calculate Thresholds (for loop)
# * Get candidates
# * Apply thresholds (for loop)

# detect-phREM
# plot-results: plot_bursts + plot_spectra + plot_iti + plot_hypno
# compute-phREM-stats

# detect-phREM
#   compute-thresholds
#   inter-trough intervals: Amp and phase
def pipeline(data, hypno, fs, nfilt=11, thr_dur=900, pplot=False):
    rem_seq = get_sequences(np.where(hypno == 5)[0])
    rem_idx = [(start * fs, (end+1) * fs) for start, end in rem_seq]

    rem_idx = ensure_duration(rem_idx, min_dur=3)
    if len(rem_idx) == 0:
        raise ValueError("No REM epochs greater than min_dur.")

    # get REM segments
    rem_epochs = get_segments(rem_idx, data)

    # Combine the REM indices with the corresponding downsampled segments
    rem = {seq:seg for seq, seg in zip(rem_seq, rem_epochs)}

    trdiff_list = []
    rem_eeg = []
    eeg_seq = {}
    sdiff_seq = {}
    tridx_seq = {}

    filt = np.ones((nfilt,))
    filt = filt / filt.sum()

    for idx in rem:
        start, end = idx

        epoch = rem[idx]
        amp, phase = _compute_hilbert(epoch, fs, pass_type="bandpass", f_range=(5, 12), remove_edges=False)

        # amplitude of the entire REM sleep
        rem_eeg += amp

        # trough indices
        tridx = _detect_troughs(phase, -3)

        # differences between troughs
        trdiff = np.diff(tridx)

        # smoothed trough differences
        sdiff_seq[idx] = np.convolve(trdiff, filt, 'same')

        # dict of trough differences for each REM period
        tridx_seq[idx] = tridx

        eeg_seq[idx] = amp
    
        # differences between troughs
        trdiff_list += list(trdiff)

        
    rem_eeg = np.array(rem_eeg)
    trdiff = np.array(trdiff_list)
    trdiff_sm = np.convolve(trdiff, filt, 'same')

    # potential candidates for phasic REM:
    # the smoothed difference between troughs is less than
    # the 10th percentile:
    thr1 = np.percentile(trdiff_sm, 10)
    # the minimum smoothed difference in the candidate phREM is less than
    # the 5th percentile
    thr2 = np.percentile(trdiff_sm, 5)
    # the peak amplitude is larger than the mean of the amplitude
    # of the REM EEG.
    thr3 = rem_eeg.mean()

    phasicREM = {rem_idx:[] for rem_idx in rem.keys()}

    for rem_idx in tridx_seq:
        rem_start, rem_end = rem_idx
        offset = rem_start * fs

        # trough indices
        tridx = tridx_seq[rem_idx]

        # smoothed trough interval
        sdiff = sdiff_seq[rem_idx]

        # amplitude of the REM epoch
        eegh = eeg_seq[rem_idx]

        # get the candidates for phREM
        cand_idx = np.where(sdiff <= thr1)[0]
        cand = get_sequences(cand_idx)

        for start, end in cand:
            # Duration of the candidate in milliseconds
            dur = ( (tridx[end]-tridx[start]+1)/fs ) * 1000
            if dur < thr_dur:
                continue # Failed Threshold 1
            
            min_sdiff = np.min(sdiff[start:end])
            if min_sdiff > thr2:
                continue # Failed Threshold 2
            
            mean_amp =  np.mean(eegh[tridx[start]:tridx[end]+1])
            if mean_amp < thr3:
                continue # Failed Threshold 3
            
            t_a = tridx[start] + offset
            t_b = np.min((tridx[end] + offset, rem_end * fs))

            ph_idx = (t_a, t_b+1)
            phasicREM[rem_idx].append(ph_idx)

    phasic = []
    for rem_idx in phasicREM:
        phasic += phasicREM[rem_idx]
    
    if pplot:
        
        p_t = create_hypnogram(phasic, len(data))

        t_vec = np.arange(0, len(data)/fs, 1/fs)

        nsr_seg = 1
        perc_overlap = 0.8
        vm = 3000

        cmap = plt.cm.jet
        my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 5)

        freq, t, SP = spectrogram(data, fs=fs, window='hann', 
                                  nperseg=int(nsr_seg * fs), 
                                  noverlap=int(nsr_seg * fs * perc_overlap))
        dt = t[1] - t[0]
        ifreq = np.where(freq <= 20)[0]

        gamma = (50, 90)
        df = freq[1] - freq[0]
        igamma = np.where((freq >= gamma[0]) & (freq <= gamma[1]))[0]
        pow_gamma = SP[igamma,:].sum(axis=0) * df

        #%% Plotting
        fig = plt.figure(figsize=(12,6), layout='tight')
        fig.suptitle(title, fontsize=12)
        axs = fig.subplot_mosaic([["states"],
                                  ["lfp"],
                                  ["phasic"],
                                  ["iti"],
                                  ["spectrogram"],
                                  ["gamma"]], sharex=True,
                                 gridspec_kw = {'height_ratios':[1, 8, 1, 8, 8, 8],
                                                'hspace':0.05}
                                 )
        tmp = axs["states"].pcolorfast(t_vec, [0, 1], np.array([hypno]), vmin=1, vmax=5)
        tmp.set_cmap(my_map)
        _despine_axes(axs["states"])

        axs["lfp"].plot(t_vec, data, color='k')

        axs["spectrogram"].pcolorfast(t, freq[ifreq], SP[ifreq, :], vmin=0, vmax=vm, cmap='jet')
        axs["spectrogram"].set_ylabel("Freq. (Hz)")

        axs["phasic"].set_ylabel("Phasic")
        axs["phasic"].step(t_vec, p_t, c='b')
        _despine_axes(axs["phasic"])

        for rem_idx in phasicREM:    
            rem_start, rem_end = rem_idx
            rem_start, rem_end = rem_start*fs, (rem_end+1)*fs

            tridx = tridx_seq[rem_idx] 
            sdiff = sdiff_seq[rem_idx]
            eegh = eeg_seq[rem_idx]

            tridx = (tridx + rem_start)/fs

            axs["lfp"].axvspan(rem_idx[0], rem_idx[1], facecolor=[0.8, 0.8, 0.8], alpha=0.3)
            for start, end in phasicREM[rem_idx]:
                axs["lfp"].plot(t_vec[start:end], data[start:end], color='r')

            axs["iti"].plot(tridx[:-1], sdiff, drawstyle="steps-pre", color='k')
            axs["lfp"].plot(t_vec[rem_start:rem_end], eegh, 'y', '--')
            axs["lfp"].plot([t_vec[rem_start], t_vec[rem_end]], [thr3, thr3], 'r', '--')

        axs["iti"].axhline(y=thr1, color='r', linestyle='--')
        axs["iti"].axhline(y=thr2, color='y', linestyle='--')
        axs["iti"].set_ylabel("ITI")

        axs["gamma"].plot(t, pow_gamma, '.-')
        axs["gamma"].set_ylabel(r'$\gamma$')

        axs["lfp"].set_ylabel("LFP")