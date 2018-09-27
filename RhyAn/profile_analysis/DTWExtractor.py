import numpy as np
import scipy.signal as sp
import librosa
from dtw import dtw
from ..AudioData import AudioData
from ..OnSetProfile import OnSetProfile

class DTWExtractor: 
    audio_data = None

    def __init__(self, audio_data):
        self.audio_data = audio_data

    @staticmethod
    def sliceProfile(profile, beats):
        return list(
            filter(lambda x: (x >= beats[0] and x <= beats[1]), profile.frames)
        )

    @staticmethod
    def slicer(profiles ,beats):
        return list(
            map(lambda x: DTWExtractor.sliceProfile(x, beats), profiles)
        )

    @staticmethod  
    def formatClusters(profiles):
        clusters = []
        for index,compo in enumerate(profiles):
            for frame in compo:
                clusters.append([frame,index])
        return np.array(clusters)
    
    @staticmethod
    def profileDTW(x_,y_):
        dists = []
        for i, (x, y) in enumerate(zip(x_, y_)):
            if not(len(x) is 0 or len(y) is 0):
                ref = np.array(DTWExtractor.formatClusters([x])).reshape(-1, 1)
                target = np.array(DTWExtractor.formatClusters([y])).reshape(-1, 1)
                dist, cost, acc, path = dtw(ref, target, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
                dists.append(dist)
        if len(dists) is not 0:
            weight = list(map(lambda x :x*5 ,list(range(1,len(dists)+1))[::-1]))
            return np.average(dists,weights=weight)
        else:
            return -1

    def averageBars(self):
        output = {}

        bar_beats = self.audio_data.beats[::32]
        step_beats = self.audio_data.beats[::16]

        for bar_i,bar in enumerate(bar_beats[0:-1]):
            slice_ref = DTWExtractor.slicer(self.audio_data.getProfiles(),[bar,bar_beats[bar_i+1]])
            bar_dist = {}
            for beat_i,beat in enumerate(step_beats):
                cur_i = beat_i
                if cur_i+3 >= len(step_beats): break
                slice_ = DTWExtractor.slicer(self.audio_data.getProfiles(),[beat,step_beats[cur_i+2]])
                distance = DTWExtractor.profileDTW(slice_ref, slice_)
                bar_dist[beat,step_beats[cur_i+2]] = distance

            avg = np.average(list(bar_dist.values()))
            output[avg] = (bar,bar_beats[bar_i+1])
        return output

    def getCommonBar(self,send_key=False):
        output = self.averageBars()
        key = np.amin(list(output.keys()))
        slice_ = DTWExtractor.slicer(self.audio_data.getProfiles(),output[key])
        common_bar = list(map(lambda x : OnSetProfile(x,[]),slice_))
        return (common_bar,output[key]) if send_key else common_bar