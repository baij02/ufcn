
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class CustomDataset(Dataset):
    def __init__(self, train = False):
        # self.imgs_path = "../newData/"
        # file_list = glob.glob(self.imgs_path + "*")
        # print(file_list)
        # self.data = []
        # for class_path in file_list:
        #     class_name = class_path.split("/")[-1]
        #     for img_path in glob.glob(class_path + "/*.png"):
        #         self.data.append([img_path, class_name])
        # print(self.data)
        # self.class_map = {"Normal" : 0, "Abnormal": 1}
        self.img_dim = (1024, 1024)

        self.data = []
        self.segDic = {}
        segFiles = glob.glob('seg/**.png')
        for i in segFiles:
            name = i.split('/')[-1]
            self.segDic[name] = i

        metaFile = "FFS_test_Meta.csv"
            
        with open(metaFile) as f:
            for line in f:
                arr = line.split(',')
                self.data.append([arr[0].replace('\n', ''), arr[1].replace('\n', ''),arr[2].replace('\n', '')])

        random.shuffle(self.data)
        random.shuffle(self.data)
        self.fileName = ''

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        InfoData = self.data[idx]
        #print(InfoData)
        imgP1 = InfoData[0].replace('\n',"")

        self.fileName = imgP1.split('/')[-1]
        imgP2 = InfoData[1].replace('\n',"")
        class_name = InfoData[2].replace('\n',"")
        #print(imgP1)
        #img_path, class_name = self.data[idx]
        img1 = cv2.imread(imgP1, 0)
        img2 = cv2.imread(imgP2, 0)

        img1 = cv2.resize(img1, self.img_dim)
        img2 = cv2.resize(img2, self.img_dim)
        class_id = int(class_name)

        # 1 abnormal
        if class_id == 1:
            label = cv2.imread(self.segDic[imgP1.split('/')[-1]], 0)
            label = cv2.resize(label, self.img_dim)
            label = np.expand_dims(label, axis=0)           
            label = torch.from_numpy(label)
           # label = label.permute(2, 0, 1)
        else:
            label = torch.rand(1024, 1024)
            label = np.expand_dims(label, axis=0)           
            label = torch.from_numpy(label)
            #label = label.permute(2, 0, 1)
        
        img1 = np.expand_dims(img1, axis=0)
        img_tensor1 = torch.from_numpy(img1)
        #img_tensor1 = img_tensor1.permute(2, 0, 1)

        img2 = np.expand_dims(img2, axis=0)
        img_tensor2 = torch.from_numpy(img2)
        #img_tensor2 = img_tensor2.permute(2, 0, 1)

        class_id = torch.tensor([class_id])
        return img_tensor1 /255., img_tensor2 /255., class_id, label

    def get_fileName():
        return self.fileName
# if __name__ == "__main__":
#     dataset = CustomDataset()
#     data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

#     for img1s, img2s, labels, label in data_loader:
#         print("Batch of images has shape: ",img1s.shape)
#         print("Batch of labels has shape: ", labels.shape)
#         print("label", label.shape)