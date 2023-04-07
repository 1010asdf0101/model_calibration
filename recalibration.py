import numpy as np
import torch
import matplotlib.pyplot as plt

class HistogramBinning():
    def __init__(self, logits, labels, n_bins=15):
        if type(logits) == torch.Tensor: self.logits = logits.detach().cpu().numpy()
        elif type(logits) == np.ndarray: self.logits = logits
        else: self.logits = np.array(logits)
        if type(labels) == torch.Tensor: self.labels = labels.detach().cpu().numpy()
        elif type(labels) == np.ndarray: self.labels = labels
        else: self.labels = np.array(labels)
        self.n_bins = n_bins
        self.classes = len(self.logits[0])
        
    def fit_histogram(self):
        Bins = np.zeros((self.classes, self.n_bins, 2), dtype=np.float32)
        for i, x in enumerate(self.logits):
            sm = np.exp(x) / np.sum(np.exp(x)) # softmax 
            for j, p in enumerate(sm):
                bidx = 0
                while p > (bidx+1)/self.n_bins: bidx+=1
                Bins[j][bidx][0] += 1*int(self.labels[i] == j) # indicator funtion
                Bins[j][bidx][1] += 1
        self.histBin = np.array([[(Bins[i, j, 0]/Bins[i, j, 1] if Bins[i, j, 1]>0 else 0) for j in range(self.n_bins)] for i in range(self.classes)], dtype = np.float32)        
        return self.histBin
    
    def __calibrate(self, x):
        sm = np.exp(x) / np.sum(np.exp(x)) # softmax
        ret = np.zeros(self.classes, dtype = np.float32)
        for j, p in enumerate(sm):
            bidx = 0
            while p > (bidx+1)/self.n_bins: bidx+=1
            ret[j] = self.histBin[j][bidx]
        return ret
    
    def calibrate(self, x=None):
        if x is None:
            return np.array([self.__calibrate(self.logits[i]) for i in range(self.logits.shape[0])], dtype = np.float32)
        if type(x) == torch.Tensor: dt = x.detach().cpu().numpy()
        elif type(x) == np.ndarray: dt = x
        else :dt = np.array(x)
        if len(dt.shape) == 1: return self.__calibrate(dt)
        return np.array([self.__calibrate(dt[i]) for i in range(dt.shape[0])], dtype = np.float32)

    def make_average_bins(self):
        Bins = np.zeros((self.n_bins, 2), dtype=np.float32)
        Q = self.calibrate()
        for i, x in enumerate(Q):
            y = np.argmax(x)
            q = x.max()/x.sum()
            bidx = 0
            while q > (bidx+1)/self.n_bins: bidx+=1
            Bins[bidx][0] += 1*int(self.labels_np[i] == np.argmax(x)) # indicator funtion
            Bins[bidx][1] += 1
        return np.array([(Bins[i, 0]/Bins[i, 1] if Bins[i, 1]>0 else 0) for i in range(self.n_bins)], dtype = np.float32)
    
    def ECE(self):
        Bins = self.make_average_bins()
        N = Bins[:, 1].sum()
        weights = np.array([])
        delta = 1.0/self.n_bins
        x = np.arange(0,1,delta)
        mid = np.linspace(delta/2,1-delta/2,self.n_bins)
        
        #error = np.abs(np.subtract(mid, ))
        

def draw_rel_diagram(bin_acc, n_bins = 15, title = None):
        #computations
        delta = 1.0/n_bins
        x = np.arange(0,1,delta)
        mid = np.linspace(delta/2,1-delta/2,n_bins)
        error = np.abs(np.subtract(mid, bin_acc))

        plt.rcParams["font.family"] = "serif"
        #size and axis limits
        plt.figure(figsize=(3,3))
        plt.xlim(0,1)
        plt.ylim(0,1)
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
        #plot bars and identity line
        plt.bar(x, bin_acc, color = 'b', width=delta,align='edge',edgecolor = 'k',label='Outputs',zorder=5)
        plt.bar(x, error, bottom=np.minimum(bin_acc,mid), color = 'mistyrose', alpha=0.5, width=delta,align='edge',edgecolor = 'r',hatch='/',label='Gap',zorder=10)
        ident = [0.0, 1.0]
        plt.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
        #labels and legend
        plt.ylabel('Accuracy',fontsize=13)
        plt.xlabel('Confidence',fontsize=13)
        plt.legend(loc='upper left',framealpha=1.0,fontsize='medium')
        if title is not None:
            plt.title(title,fontsize=16)
        plt.tight_layout()

        return plt
