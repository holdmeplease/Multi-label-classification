from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms
import torch

Transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])

class myDataSet(data.Dataset):
    def __init__(self, root, istest, transfrom):
        self.root = root
        self.data_txt = open('annotations.txt', 'r')
        self.istest = istest
        self.transform = transfrom
        self.imgs = []
        for line in self.data_txt:
            line = line.rstrip()
            words = line.split()
            if self.istest:
                if words[0][0:4] == '2007' or words[0][0:4] == '2008':
                    label_cur = []
                    for i in range(1, len(words)):
                        label_cur.append(int(words[i]))
                    self.imgs.append([words[0], label_cur])
            else:
                if not (words[0][0:4] == '2007' or words[0][0:4] == '2008'):
                    label_cur = []
                    for i in range(1, len(words)):
                        label_cur.append(int(words[i]))
                    self.imgs.append([words[0], label_cur])
                    
    def __getitem__(self, index):
        cur_img = Image.open(self.root + self.imgs[index][0] + '.jpg')
        data_once = self.transform(cur_img)
        label_once = self.imgs[index][1]
        return data_once, torch.Tensor(label_once)
    
    def __len__(self):
        return len(self.imgs)

trainData = myDataSet('JPEGImages/', 0, Transform)
testData = myDataSet('JPEGImages/' ,1, Transform)

print('trainData', len(trainData))
print('testData', len(testData))
