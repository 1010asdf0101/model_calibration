import torchvision.transforms as trs
from torch.utils.data import DataLoader
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import timm
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.nn.init as init
import torchvision
import torchvision as tv
import torchvision.transforms as transforms
import calibration_library.metrics as metrics
import calibration_library.recalibration as recalibration
import calibration_library.visualization as visualization
from typing import List, Callable, Tuple
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path

class BaseSet(Dataset):
    def __init__(self,
                 paths: List[str],
                 labels: List[int],
                 transforms: Callable = None,
                 ) -> None:
        super().__init__()
        self.paths = paths
        self.labels = labels
        self.transforms = transforms
        
    def __getitem__(self,
                    index,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[index]
        label = self.labels[index]
        
        img = cv2.imread(str(path))
        if self.transforms:
            img = self.transforms(img)
        
        label = torch.tensor(label)
        img = img.to(device)
        label = label.to(device)
        return img, label    
    def __len__(self):
        return len(self.paths)

device = 'cuda:1'

PATH = 'data/resnet18_total(1+2)_best.pt'
trained = timm.create_model('resnet18', pretrained=False, num_classes = 6)
trained.load_state_dict(torch.load(PATH, map_location=device))
trained.to(device)
ROOT = Path('/home/shawnman99/calibration')
DATA_DIR = ROOT / "data"
CLASS_NAMES = ['crack', 'ddul', 'imul', 'ok', 'scratch', 'void']
test_paths = [p for p in (DATA_DIR / f'total(1+2)_split/valid').glob('*/*')
                if p.suffix in ['.png', '.jpg'] and p.parent.name in CLASS_NAMES]
test_labels = [CLASS_NAMES.index(p.parent.name) for p in test_paths]
testset = BaseSet(test_paths, test_labels, trs.Compose([
    trs.ToTensor(),
    trs.Resize((384,384))
]))
testloader = DataLoader(testset, batch_size=1, shuffle=False)
#trained.eval()
#with torch.no_grad():
    #print(trained())
model = recalibration.ModelWithTemperature(trained)
model.set_temperature(testloader)

