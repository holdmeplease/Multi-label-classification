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
from visdom import Visdom
import os
from tensorboardX import SummaryWriter

torch.cuda.set_device(3)

parser = argparse.ArgumentParser(description='VGG-16 Input:BatchSize initial LR EPOCH')
parser.add_argument('--test','-t', action = 'store_true',
 help='set test mode')
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
print('model_path:',model_path)
print('batch_size:',BATCH_SIZE)
print('initial LR:',LR)
print('epoch:',EPOCH)

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

#viz=Visdom(use_incoming_socket=False)
#viz.line([0.],[0.],win='train_loss',opts=dict(title='train_loss'))
writer = SummaryWriter('VGG_16')

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
#global_step=0

if not args.test:
    # Loss  Optimizer Scheduler
    cost = nn.BCELoss(weight=None, size_average=True)#input:Float target:Float
    optimizer = torch.optim.Adam(vgg_16.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Train the model
    for epoch in range(EPOCH):
        scheduler.step(epoch)
        for i, (images, labels) in enumerate(trainLoader):
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
            #viz.line([loss.item()],[global_step],win='train_loss',update='append')
            #global_step+=1
            #validating
            if (i+1) % 100 == 0 :
                print ('Epoch [%d/%d], Iter[%d/%d] Loss %.9f' %
                    (epoch+1, EPOCH, i+1, len(trainData)//BATCH_SIZE, loss.item()))
        
        correct = 0
        total = 0
        vec_1=torch.Tensor(1,20).zero_()
        vec_2=torch.Tensor(1,20).zero_()
        for images, labels in testLoader:
            images = Variable(images).cuda()
            labels= Variable(labels).cuda()
            outputs = vgg_16(images)
            outputs=torch.sigmoid(outputs)
            predicted = outputs.data>=0.5
            vec_1 += (predicted.float() == labels).cpu().float().sum(0) #correct_num
            vec_2 += labels.cpu().sum(0)#appear_num
            #equal to predicted=outputs.data>=0
            total += labels.size(0)*labels.size(1)
            correct += (predicted.float() == labels).sum()

        vec_1=vec_1.float()/len(testData)
        vec_2=vec_2.float()/vec_2.sum()
        print('TestSet Class Accuracy:',vec_1)
        print('Epoch [%d/%d]:Test Accuracy of the model on the test images(mAcc): %.4f %%' % (epoch+1,EPOCH,100 * float(correct) / float(total)))
        print('Epoch [%d/%d]:Test Accuracy of the model on the test images(wAcc): %.4f %%' % (epoch+1,EPOCH,100 * (vec_1*vec_2).sum()))
        writer.add_scalar('Train/loss', loss.item(),epoch)
        writer.add_scalar('Train/mAcc', float(correct) / float(total),epoch)       
        writer.add_scalar('Train/wAcc', (vec_1*vec_2).sum(),epoch)
        
        torch.save(vgg_16.state_dict(), os.path.join(model_path, 'vgg_16.pkl'))
    # Save the Trained Model    
    writer.close()
    torch.save(vgg_16.state_dict(), os.path.join(model_path, 'vgg_16.pkl'))
else:
    # Test the model
    vgg_16.eval()
    correct = 0
    total = 0
    vec_1=torch.Tensor(1,20).zero_()
    vec_2=torch.Tensor(1,20).zero_()
    TP=torch.Tensor(1,20).zero_()
    FP=torch.Tensor(1,20).zero_()
    FN=torch.Tensor(1,20).zero_()    
    for images, labels in trainLoader:
        images = Variable(images).cuda()
        labels= Variable(labels).cuda()
        outputs = vgg_16(images)
        outputs=torch.sigmoid(outputs)
        predicted = outputs.data>=0.5
        total += labels.size(0)*labels.size(1)
        vec_1 += (predicted.float() == labels).cpu().float().sum(0) #correct_num
        vec_2 += labels.cpu().sum(0)#appear_num
        TP+=((predicted.float() == labels).cpu().float()*(labels==1).cpu().float()).sum(0)
        FP+=((predicted.float() != labels).cpu().float()*(labels==1).cpu().float()).sum(0)
        FN+=((predicted.float() != labels).cpu().float()*(labels==0).cpu().float()).sum(0)
        correct += (predicted.float() == labels).sum()
    vec_1=vec_1.float()/len(trainData)
    vec_2=vec_2.float()/vec_2.sum()
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1=2*precision*recall/(precision+recall)
    F1_mean=F1.sum()/len(F1[0])
    print('TrainSet Class Accuracy:',vec_1)
    #viz.images(images.view(3,224,224),win='pic')
    #viz.text(str(labels.detach().cpu().numpy()),win='true_label',opts=dict(title='true_label'))
    #viz.text(str(predicted.detach().cpu().numpy()),win='predicted_label',opts=dict(title='predicted_label'))
    print('Test Accuracy of the model on the train images(mAcc): %.4f %%' % (100 * float(correct) / float(total)))
    print('Test Accuracy of the model on the train images(wAcc): %.4f %%' % (100 * (vec_1*vec_2).sum()))
    print('mean F1 score of the model on the train images: %.4f ' % (F1_mean))
    correct = 0
    total = 0
    vec_1=torch.Tensor(1,20).zero_()
    vec_2=torch.Tensor(1,20).zero_()
    TP=torch.Tensor(1,20).zero_()
    FP=torch.Tensor(1,20).zero_()
    FN=torch.Tensor(1,20).zero_()
    for images, labels in testLoader:
        images = Variable(images).cuda()
        labels= Variable(labels).cuda()
        outputs = vgg_16(images)
        outputs=torch.sigmoid(outputs)
        predicted = outputs.data>=0.5
        vec_1 += (predicted.float() == labels).cpu().float().sum(0) #correct_num
        vec_2 += labels.cpu().sum(0)#appear_num
        #equal to predicted=outputs.data>=0
        total += labels.size(0)*labels.size(1)
        TP+=((predicted.float() == labels).cpu().float()*(labels==1).cpu().float()).sum(0)
        FP+=((predicted.float() != labels).cpu().float()*(labels==1).cpu().float()).sum(0)
        FN+=((predicted.float() != labels).cpu().float()*(labels==0).cpu().float()).sum(0)
        correct += (predicted.float() == labels).sum()
    #viz.images(images.view(3,224,224),win='pic')
    #viz.text(str(labels.detach().cpu().numpy()),win='true_label',opts=dict(title='true_label'))
    #viz.text(str(predicted.detach().cpu().numpy()),win='predicted_label',opts=dict(title='predicted_label'))
    vec_1=vec_1.float()/len(testData)
    vec_2=vec_2.float()/vec_2.sum()
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1=2*precision*recall/(precision+recall)
    F1_mean=F1.sum()/len(F1[0])
    print('TestSet Class Accuracy:',vec_1)
    print('Test Accuracy of the model on the test images(mAcc): %.4f %%' % (100 * float(correct) / float(total)))
    print('Test Accuracy of the model on the test images(wAcc): %.4f %%' % (100 * (vec_1*vec_2).sum()))
    print('mean F1 score of the model on the test images: %.4f ' % (F1_mean))
print(summary(vgg_16,(3,224,224)))
