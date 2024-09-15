from .detect_phasic import compute_thresholds, get_rem_epochs, get_phasic_candidates, is_valid_phasic
from .utils import get_start_end

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pynapple as nap

from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram

class PhasicTonic:
    """
    A class for analyzing phasic and tonic REM sleep periods in EEG data.
    """

    def __init__(self, fs: float, thr_dur: float):
        """
        Initialize the PhasicTonic object.

        Parameters
        ----------
        fs : (float)
            Sampling frequency of the EEG data.
        thr_dur : (float)
            Threshold duration for phasic REM detection.
        """
        self.fs = fs
        self.thr_dur = thr_dur
        self.eeg = None
        self.t = None
        self.hypno = None
        self.rem_interval = None
        self.phasic_interval = None
        self.tonic_interval = None
        self.thresholds = None
        self.trough_idx_seq = None
        self.smooth_difference_seq = None
        self.eeg_seq = None
        
    def fit(self, eeg: np.ndarray, hypno: np.ndarray) -> dict:
        """
        Fit the model to the EEG and hypnogram data.

        Parameters
        ----------
        eeg : np.ndarray
            EEG time series data.
        hypno : np.ndarray
            Hypnogram data.

        Returns
        -------
        Dict[Tuple[int, int], list]
            Dictionary containing phasic REM intervals for each REM epoch.
        """
        
        self._prepare_data(eeg, hypno)
        
        # Extract REM epochs
        rem_epochs = get_rem_epochs(eeg=eeg, hypno=hypno, fs=self.fs, min_dur=3)
        
        # Compute thresholds for detecting phasic REM
        self.thresholds, self.trough_idx_seq, self.smooth_difference_seq, self.eeg_seq = compute_thresholds(rem_epochs, self.fs)
        
        phasic_rem = {rem_idx: [] for rem_idx in rem_epochs}
        thr1, thr2, thr3 = self.thresholds

        for rem_idx, trough_idx in self.trough_idx_seq.items():
            rem_start, rem_end = rem_idx
            offset = rem_start * self.fs
            smooth_difference, eegh = self.smooth_difference_seq[rem_idx], self.eeg_seq[rem_idx]

            candidates = get_phasic_candidates(smooth_difference, trough_idx, thr1, self.thr_dur, self.fs)

            for start, end in candidates:
                if is_valid_phasic(start, end, smooth_difference, eegh, trough_idx, thr2, thr3):
                    t_a = trough_idx[start] + offset
                    t_b = min(trough_idx[end] + offset, rem_end * self.fs)
                    phasic_rem[rem_idx].append((t_a, t_b + 1))

        self._create_interval_sets(phasic_rem)
        return phasic_rem
    
    def _prepare_data(self, eeg: np.ndarray, hypno: np.ndarray):
        """
        Prepare the input data for analysis.
        """
        self.t = np.arange(0, len(eeg) / self.fs, 1 / self.fs)
        self.eeg = nap.Tsd(t=self.t, d=eeg)
        self.hypno = hypno
        rem_start, rem_end = get_start_end(hypno, 5)
        self.rem_interval = nap.IntervalSet(rem_start, rem_end)

    def _create_interval_sets(self, phasic_rem: dict):
        """
        Create interval sets for phasic and tonic REM periods.
        """
        start, end = [], []
        for rem_idx in phasic_rem:
            for s, e in phasic_rem[rem_idx]:
                start.append(s / self.fs)
                end.append(e / self.fs)
        self.phasic_interval = nap.IntervalSet(start, end)
        self.tonic_interval = self.rem_interval.set_diff(self.phasic_interval)
    
    def plot(self):
        """
        Plot the results of the analysis.
        """
        fig, axs = self._create_plot_layout()
        self._plot_sleep_states(axs["states"])
        self._plot_lfp(axs["lfp"])
        self._plot_spectrogram(axs["spectrogram"])
        self._plot_phasic_rem(axs["phasic"])
        self._plot_iti(axs["iti"])
        self._plot_gamma(axs["gamma"])
        plt.show()

    def _create_plot_layout(self):
        """Create the plot layout."""
        fig = plt.figure(figsize=(12, 6), layout='constrained')
        axs = fig.subplot_mosaic([
            ["states"],
            ["lfp"],
            ["phasic"],
            ["iti"],
            ["spectrogram"],
            ["gamma"]
        ], sharex=True, gridspec_kw={'height_ratios': [1, 8, 1, 8, 8, 8], 'hspace': 0.05})
        return fig, axs

    def _plot_sleep_states(self, ax):
        """Plot sleep states."""
        colors = [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]]
        my_map = LinearSegmentedColormap.from_list('brs', colors, N=5)
        tmp = ax.pcolorfast(self.t, [0, 1], np.array([self.hypno]), vmin=1, vmax=5)
        tmp.set_cmap(my_map)
        self._despine_axes(ax)

    def _plot_lfp(self, ax):
        """Plot LFP data."""
        ax.plot(self.t, self.eeg, color='k')
        ax.set_ylabel("LFP")
        for epoch in self.rem_interval:
            rem_start, rem_end = int(epoch["start"].item()), int(epoch["end"].item())
            ax.axvspan(rem_start, rem_end, facecolor=[0.7, 0.7, 0.8], alpha=0.4)
            ax.plot(self.t[rem_start*self.fs:(rem_end+1)*self.fs], 
                    self.eeg_seq[(rem_start, rem_end)], 'y', '--')
            ax.plot([self.t[rem_start*self.fs], self.t[(rem_end+1)*self.fs]], 
                    [self.thresholds[2], self.thresholds[2]], 'r', '--')
        [ax.plot(self.eeg.restrict(self.phasic_interval[i]), color='r') 
         for i in range(len(self.phasic_interval))]

    def _plot_spectrogram(self, ax):
        """Plot spectrogram."""
        nsr_seg, perc_overlap, vm = 1, 0.8, 3000
        freq, t, SP = spectrogram(self.eeg, fs=self.fs, window='hann', 
                                  nperseg=int(nsr_seg * self.fs), 
                                  noverlap=int(nsr_seg * self.fs * perc_overlap))
        ifreq = np.where(freq <= 20)[0]
        ax.pcolorfast(t, freq[ifreq], SP[ifreq, :], vmin=0, vmax=vm, cmap='hot')
        ax.set_ylabel("Freq. (Hz)")

    def _plot_phasic_rem(self, ax):
        """Plot phasic REM as spikes."""
        ax.set_ylabel("Phasic")
        ax.eventplot((self.phasic_interval["end"] + self.phasic_interval["start"]) / 2)
        self._despine_axes(ax)

    def _plot_iti(self, ax):
        """Plot inter-trough intervals."""
        for epoch in self.rem_interval:
            rem_start, rem_end = int(epoch["start"].item()), int(epoch["end"].item())
            tridx = self.trough_idx_seq[(rem_start, rem_end)]
            sdiff = self.smooth_difference_seq[(rem_start, rem_end)]
            tridx = (tridx + rem_start * self.fs) / self.fs
            ax.plot(tridx[:-1], sdiff, drawstyle="steps-pre", color='k')
        ax.axhline(y=self.thresholds[0], color='r', linestyle='--')
        ax.axhline(y=self.thresholds[1], color='y', linestyle='--')
        ax.set_ylabel("ITI")

    def _plot_gamma(self, ax):
        """Plot gamma power."""
        freq, t, SP = spectrogram(self.eeg, fs=self.fs, window='hann', 
                                  nperseg=int(self.fs), 
                                  noverlap=int(self.fs * 0.8))
        gamma = (50, 90)
        df = freq[1] - freq[0]
        igamma = np.where((freq >= gamma[0]) & (freq <= gamma[1]))[0]
        pow_gamma = SP[igamma,:].sum(axis=0) * df
        ax.plot(t, pow_gamma, '.-')
        ax.set_ylabel(r'$\gamma$')
    
    def compute_stats(self) -> pd.DataFrame:
        """
        Compute statistics for phasic and tonic REM periods.

        Returns
        -------
        pd.DataFrame
            DataFrame containing computed statistics.
        """
        stats = {
            "rem_start": [], "rem_end": [], "state": [],
            "num_bouts": [], "mean_duration": [],
            "total_duration": [], "percent_of_rem": []
        }

        for rem_idx in self.rem_interval:
            phasic = rem_idx.intersect(self.phasic_interval)
            tonic = rem_idx.intersect(self.tonic_interval)

            for state, intervals in [("phasic", phasic), ("tonic", tonic)]:
                self._compute_interval_stats(rem_idx, state, intervals, stats)

        return pd.DataFrame(stats)

    def _compute_interval_stats(self, rem_idx, state, intervals, stats):
        """
        Compute statistics for a given interval.
        """
        num_bouts = len(intervals)
        durations = np.diff(intervals, 1)
        total_duration = np.sum(durations)
        percent_of_rem = total_duration / rem_idx.tot_length()

        stats["rem_start"].append(int(rem_idx["start"].item()))
        stats["rem_end"].append(int(rem_idx["end"].item()))
        stats["state"].append(state)
        stats["num_bouts"].append(num_bouts)
        stats["mean_duration"].append(durations.mean())
        stats["total_duration"].append(total_duration)
        stats["percent_of_rem"].append(percent_of_rem)
    
    def get_phasic(self) -> nap.IntervalSet:
        """
        Get the phasic REM intervals.

        Returns
        -------
        nap.IntervalSet
            Phasic REM intervals.
        """
        return self.phasic_interval

    def get_tonic(self) -> nap.IntervalSet:
        """
        Get the tonic REM intervals.

        Returns
        -------
        nap.IntervalSet
            Tonic REM intervals.
        """
        return self.tonic_interval
    
    @staticmethod
    def _despine_axes(ax):
        """Remove top and right spines from the axes."""
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
