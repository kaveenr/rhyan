import numpy as np
import scipy.signal as sp
import scipy.spatial.distance as sp_dist
import librosa

class MedianNMF:
    y, sr = None,None
    n_components = None

    def __init__(self,y,sr,n_components = 5):
        self.y, self.sr = y,sr
        self.n_components = n_components
    
    def decompose(self):
        #filter out precussive parts
        hpss_y = self.hpss()
        #Perform Short-time Fourier transform
        D = librosa.stft(hpss_y)
        # Separate the magnitude and phase
        S, phase = librosa.magphase(D)
        #NMF decompose to components
        components, activations = self.decomposeNMF(hpss_y, S, self.n_components)
        #reconstruct and return
        return [self.reconstructComponent(
            components[:, i], activations[i], phase) for i in range(0,len(activations))]
    
    def hpss(self, margin=4.0):
        #extract precussive components through median filtering
        return librosa.effects.percussive(self.y, margin=margin)

    def decomposeNMF(self, y, magnitude,  n_components):
        # Decompose by nmf
        return librosa.decompose.decompose(magnitude, n_components, sort=True)

    def reconstructFull(self, activations, phase):
        #reconstruct all components into one signal
        D_k = components.dot(activations)
        y_k = librosa.istft(D_k * phase)
        return y_k
    
    def reconstructComponent(self, components, activation, phase):
        D_k = np.multiply.outer(components, activation)
        y_k = librosa.istft(D_k * phase)
        #filter out noise using Savitzky-Golay filter 
        component_filtered = sp.savgol_filter(y_k,11,1)
        return component_filtered

    

