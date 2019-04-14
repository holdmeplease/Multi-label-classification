import numpy
import torch
from torch import nn
import math
from torch.autograd import Variable
import torchvision.models as v_models
from torchsummary import summary
import torchvision.transforms as transforms
import torchvision as tv
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from tools import myDataSet
import os

model_path = './model_para'#dir to save para
BATCH_SIZE = 100
LR = 0.01
EPOCH = 5

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])

trainData = myDataSet(root='../data/train', label='../data/train/train_label',transform=transform)
testData = myDataSet(root='../data/test', label='../data/test/test_label',transform=transform)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True,num_workers=2)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

vgg_16 = v_models.vgg16(pretrained=False, num_classes=20)
if os.path.exists(os.path.join(model_path, 'vgg_16.pkl')):
    vgg_16.load_state_dict(torch.load(os.path.join(model_path, 'vgg_16.pkl')))
else:
    vgg_16_temp=v_models.vgg16(pretrained=True)
    pretrained_dict = vgg_16_temp.state_dict()
    modified_dict = vgg_16.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k !='classifier.6.weight' and k!='classifier.6.bias'}
    modified_dict.update(pretrained_dict)
    vgg_16.load_state_dict(modified_dict)
vgg_16.cuda()

# Loss and Optimizer
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg_16.parameters(), lr=LR)

# Train the model
for epoch in range(EPOCH):
    for i, (images, labels) in enumerate(trainLoader):
    #for images, labels in trainLoader:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = vgg_16(images)
        loss = cost(outputs, labels)
        loss.backward()
        optimizer.step()
        #validating
        if (i+1) % 100 == 0 :
            print ('Epoch [%d/%d], Iter[%d/%d] Loss. %.4f' %
                (epoch+1, EPOCH, i+1, len(trainData)//BATCH_SIZE, loss.data[0]))
        if (i+1)%1000 ==0:
            torch.save(nn.state_dict(), os.path.join(model_path, 'vgg_16.pkl'))

# Test the model
vgg_16.eval()
correct = 0
total = 0

for images, labels in testLoader:
    images = Variable(images).cuda()
    outputs = vgg_16(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(nn.state_dict(), os.path.join(model_path, 'vgg_16.pkl'))










