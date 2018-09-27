import librosa

class OnSetProfile:
    energy, frames = [], []

    def __init__(self, onset_frames, onset_e):
        self.frames, self.energy = onset_frames, onset_e

    def energy(self):
        return self.energy
    
    def frames(self):
        return self.frames
        
    def framesInTimes(self,sr):
        return librosa.frames_to_time(self.frames, sr=sr).tolist()

    #Keeping compatibity with the previous dict structure
    def __getitem__(self, key):
        return self.energy if key is "onset_e" else self.frames

    def __setitem__(self, key, value):
        if key is "onset_e":
            self.energy = value
        else:
            self.frames = value