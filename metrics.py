import numpy as np

class CalibrationMetrics(object):
    def __init__(self, num_classes):
        self.n_classes = num_classes
    
    def make_reliable_bins(self, probabilities, labels, n_bins=15):
        Bins = np.zeros((n_bins, 2), dtype=np.float32)
        cnt_class = np.zeros(self.classes, dtype=np.float32)
        for i, x in enumerate(probabilities):
            y = np.argmax(x)
            q = x.max()/x.sum()
            bidx = 0
            while q > (bidx+1)/n_bins: bidx+=1
            #print(f"{p} : {labels_np[i]}, {x}")
            Bins[bidx][0] += 1*int(labels_np[i] == y) # indicator funtion
            Bins[bidx][1] += 1
            
    def ECE(self, )