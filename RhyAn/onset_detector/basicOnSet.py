import numpy as np
import scipy.signal as sp
import scipy.spatial.distance as sp_dist
import librosa
from ..OnSetProfile import OnSetProfile

class BasicOnSet:
    components = []
    sr = None

    def __init__(self, components, sr):
        self.components = components
        self.sr = sr

    def apply(self):
        #Caclulate on set profiles for all components
        profiles = list(map(self.calculateOnSet,self.components))
        filtered_profies = list(map(lambda x : self.dropDuplicates(self.dropFrames(x)), profiles))
        return filtered_profies

    def calculateOnSet(self,component):
        #Use in-built basic librosa on-set detectors
        onset_frames = librosa.onset.onset_detect(y=component, sr=self.sr)
        onset_e = librosa.onset.onset_strength(component, sr=self.sr)
        return OnSetProfile(onset_frames,onset_e)

    def dropFrames(self, profile, frame_window=2, thrushold=0.2):
        #Calculate average energy and drop the other frames
        average = np.max(profile.energy) * thrushold
        filtered = list(filter(
            (lambda x : np.average(profile.energy[x-frame_window:x+frame_window]) > average)
        , profile.frames))
        return OnSetProfile(filtered,profile.energy)

    def dropDuplicates(self,profile,thrushold=5):
        #frames within certain thrushold window will be dropped
        component = profile.frames
        new_frames = []
        for index,frame in enumerate(component[1:]):
            frames = component[index:index+2]
            if np.diff(frames) > thrushold:
                new_frames.append(frames[1])
        return OnSetProfile(new_frames,profile.energy)