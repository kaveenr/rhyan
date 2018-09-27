import numpy as np
import scipy.signal as sp
import scipy.spatial.distance as sp_dist
import librosa

class FundementalBandPass:
    components = []
    sr = None
    nyquist_rate = None

    def __init__(self, components, sr):
        self.components = components
        self.sr = sr
        self.nyquist_rate = sr/2 

    def apply(self):
        return list(map(self.applyComponent, self.components))

    def applyComponent(self, component):
        sos = self.buildFilter(component)
        return sp.sosfilt(sos,component)

    def getFundementalFrequency(self, component):
        #Perform on set detection on frames
        frames = librosa.onset.onset_detect(y=component, sr=self.sr)
        #for each beat frame get maximum freq frame
        avgs = list(map(lambda x : np.argmax(librosa.stft(component)[..., x]),frames))
        #get the one with the most value thus fundemental frequency
        return np.max(avgs)

    def buildFilter(self, component,freq_window=5):
        fundemental_freq = self.getFundementalFrequency(component)
        #Parameters for the bandpass filter
        lowcut = fundemental_freq - freq_window
        hicut = fundemental_freq + freq_window
        ranges = (
            lowcut / self.nyquist_rate,
            hicut / self.nyquist_rate
        )
        #Make iirfilter filter
        #return sp.butter(1, ranges, 'bandpass', output='sos')
        return sp.iirfilter(1, ranges, btype='bandpass', output='sos')