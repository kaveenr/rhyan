import numpy as np
import scipy.signal as sp
import scipy.spatial.distance as sp_dist
import librosa
from ..OnSetProfile import OnSetProfile

class MergeProfiles:

    clustered_profiles = []

    def __init__(self, clusters): 
        self.clustered_profiles = clusters

    def apply(self):
        return list(map(lambda x : self.flattenCluster(x) , self.clustered_profiles))
    
    def toOnsetPair(self, filtered_frames, averaged_frames):
        return OnSetProfile(np.array(list(filtered_frames.keys())),[averaged_frames[x] for x in filtered_frames])

    def filterFrames(self, frames, average, tolerence=0.3):
        return {k:v for k,v in frames.items() if v > average * tolerence}

    def averageFrames(self, frames):
        frameCopy = frames.copy()
        for frame, strengths in frameCopy.items():
            frameCopy[frame] = np.average(strengths)
        return frameCopy

    def indexOfNearbyFrame(self, frames,frame,frame_window=5):
        frame_range = range(frame-frame_window,frame+frame_window)
        for i in frame_range:
            if i in frames:
                return i
        return None

    def flattenCluster(self, cluster,frame_window=5):
        frames = dict()
        strengths = []
        for component in cluster: 
            for frame in component["onset_frames"]:
                strength = np.average(component["onset_e"][frame-frame_window:frame+frame_window])
                strengths.append(strength)
                nearFrame = self.indexOfNearbyFrame(frames,frame)
                if nearFrame is None:
                    frames[frame] = [strength]
                else:
                    frames[nearFrame].append(strength)
        averaged_frames = self.averageFrames(frames)
        filtered_frames = self.filterFrames(averaged_frames,np.average(strengths))
        return self.toOnsetPair(filtered_frames,filtered_frames)