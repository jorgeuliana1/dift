import os, torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  
import numpy as np
import src.dataloader.transforms as trans
import pickle
import pandas as pd
from typing import Optional, Tuple

class MyDataset(Dataset):
    def __init__(self, root: str, csv_train: str, csv_test: str, train: bool = True):
        self.trainsize = (224,224)
        self.train = train
        self.root = root

        # Defining .csv file to be used
        if self.train:
            csv_name = csv_train
        else:
            csv_name = csv_test
            
        # Opening dataframe:
        self.df = pd.read_csv(csv_name, header=0)

    def __len__(self):
        return len(self.df)
    
    @property
    def labels_balance(self):
        y_series = self.df[self.y]
        v_counts = y_series.value_counts(normalize=True)
        sorted_v_counts = v_counts.sort_index()
        return np.asarray(sorted_v_counts)
    
class PadUfes20(MyDataset):
    def __init__(self, root: str, csv_train: str, csv_test: str, train: bool = True):
        super(PadUfes20, self).__init__(root, csv_train, csv_test, train)
        if self.train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                trans.RandomHorizontalFlip(),
                trans.RandomVerticalFlip(),
                trans.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        self.x = "img_id"
        self.y = "diagnostic_number"
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(index):
            index = index.tolist()
                
        img_path = os.path.join(self.root, "images", self.df.loc[index][self.x])
        img = Image.open(img_path).convert('RGB')
        img = self.transform_center(img)
        return img, int(self.df.loc[index][self.y])
    