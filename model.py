import numpy
import torch
from torch import nn
import argparse
from torch.autograd import Variable
import torchvision.models as v_models
from torchsummary import summary
import torchvision.transforms as transforms
import torchvision as tv
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from data_pre import myDataSet
import os

parser = argparse.ArgumentParser(description='Input:BatchSize initial LR EPOCH')
parser.add_argument('--model_path', type=str,default='./model_para',
 help='dir to save para')
parser.add_argument('--BATCH_SIZE', type=int,default=64,
 help='batch_size')
parser.add_argument('--LR', type=float,default=0.01,
 help='Learning Rate')
parser.add_argument('--EPOCH', type=int,default=100,
 help='epoch')
args = parser.parse_args()
model_path=args.model_path
BATCH_SIZE=args.BATCH_SIZE
LR=args.LR
EPOCH=args.EPOCH

Transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std  = [ 0.229, 0.224, 0.225 ]),
    ])

trainData = myDataSet('JPEGImages/', 0, Transform)
testData = myDataSet('JPEGImages/' ,1, Transform)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True,num_workers=3)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

vgg_16 = v_models.vgg16(pretrained=False, num_classes=20)
if os.path.exists(os.path.join(model_path, 'vgg_16.pkl')):
    vgg_16.load_state_dict(torch.load(os.path.join(model_path, 'vgg_16.pkl')))
else:
    pretrained_dict = torch.load('vgg16-397923af.pth')
    modified_dict = vgg_16.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k !='classifier.6.weight' and k!='classifier.6.bias'}
    modified_dict.update(pretrained_dict)
    vgg_16.load_state_dict(modified_dict)
vgg_16.cuda()
'''
# Loss  Optimizer Scheduler
cost = nn.BCELoss(weight=None, size_average=True)#input:Float target:Float
optimizer = torch.optim.Adam(vgg_16.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=33, gamma=0.1)

# Train the model
for epoch in range(EPOCH):
    scheduler.step(epoch)
    for i, (images, labels) in enumerate(trainLoader):
    #for images, labels in trainLoader:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = vgg_16(images)
        output_sig=torch.sigmoid(outputs)
        #print(outputs.size())
        #print(labels.size())
        loss = cost(output_sig, labels)
        loss.backward()
        optimizer.step()
        #validating
        if (i+1) % 100 == 0 :
            print ('Epoch [%d/%d], Iter[%d/%d] Loss %.4f' %
                (epoch+1, EPOCH, i+1, len(trainData)//BATCH_SIZE, loss.item()))
    torch.save(vgg_16.state_dict(), os.path.join(model_path, 'vgg_16.pkl'))
'''
# Test the model
vgg_16.eval()
correct = 0
total = 0

for images, labels in testLoader:
    images = Variable(images).cuda()
    outputs = vgg_16(images)
    outputs=torch.sigmoid(outputs)
    predicted = outputs.data>=0.5
    total += labels.size()
    print(labels.size())
    correct += (predicted.cpu().float() == labels).sum()
print(total)
print(correct)

print('Test Accuracy of the model on the test images: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(vgg_16.state_dict(), os.path.join(model_path, 'vgg_16.pkl'))










