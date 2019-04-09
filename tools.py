import os
import torch
import torch.utils.data as data
from PIL import Image

def loading(path):
    return Image.open(path).convert('RGB')
#读取单张图片

class myDataSet(data.Dataset):
    def __init__(self,root,label,transform=None,target_transform=None,loader=loading):
        #root源文件夹
        fh=open(label)
        imgs=[]
        class_names=[x for x in range(20)]
        for line in fh.readlines():
            cls=line.split()
            file_name=cls.pop(0)
            if os.path.isfile(os.path.join(root,fn)):
                imgs.append((file_name,tuple(float(v) for v in cls)))
        self.root=root
        self.imgs=imgs
        self.classes=class_names
        self.transform=transform
        self.traget_transform=target_transform
        self.loader=loader

    def __getitem__(self,index):
        fn,label=self.imgs[index]
        img=self.loader(os.path.join(self.root,fn))
        if self.transform is not None:
            img=self.transform(img)
        return img,torch.Tensor(label)
    def __len__(self):
        return len(self,imgs)

    def getName(self):
        return self.classes


