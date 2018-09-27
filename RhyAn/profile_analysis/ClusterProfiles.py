class ClusterProfiles:
    profiles = []

    def __init__(self, profiles):
        self.profiles = profiles

    def frameSimilarity(self, y, x, frame_thrushold=10):
        y_frame_index = 0
        lost_frames = 0
        found_frames = 0
        for x_frame in x:
            old_index = y_frame_index 
            found = False
            for index,y_frame in enumerate(y[y_frame_index:]):
                diff = x_frame - y_frame
                found = diff in range(-(frame_thrushold),frame_thrushold)
                if found:
                    break
                else:
                    y_frame_index = index
            found_frames+= 1 if found else 0
            if not found:
                lost_frames+=1
                y_frame_index = old_index
        return found_frames,lost_frames

    def apply(self):
        self.profiles.append(self.profiles[0])
        clusters = []
        index = 0
        while index < len(self.profiles)-1:
            similarities = []
            do_increment = True
            for count,pair in enumerate(self.profiles[index:]):
                similarity, lost = self.frameSimilarity(self.profiles[index].frames,pair.frames)
                similarities.append(similarity)
                if(similarity-lost < 0 or (index == len(self.profiles)-2 and count+index == len(self.profiles)-1)):
                    clusters.append(self.profiles[index:count+index])
                    index = count + index
                    do_increment = False
                    break
            index += 1 if do_increment else 0
        return clusters