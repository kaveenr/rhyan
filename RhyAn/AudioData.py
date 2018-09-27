from .OnSetProfile import OnSetProfile
import librosa

class AudioData(dict):

    default = {
        "url": "",
        "sr": 0,
        "artist": "",
        "title": "",
        "BPM_isolated": 0,
        "n_clusters": 0,
        "profiles": [],
        "BPM_overall": 0,
        "beats": [],
    }

    def __init__(self, data=None):
        super(AudioData, self).__init__(data if data else self.default)

    def __getattr__(self, key):
        return self.get(key)

    def getProfiles(self):
        return list(
            map(lambda x: OnSetProfile(x, []), self["profiles"])
        )

    def getProfileTimecode(self):
        return list(
            map(lambda x: librosa.frames_to_time(x, self["sr"]).tolist(), self["profiles"])
        )
    
    def getBeatTimecode(self):
        return librosa.frames_to_time(self['beats'], self["sr"]).tolist()
    