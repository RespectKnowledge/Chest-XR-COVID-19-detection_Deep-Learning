# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 18:48:23 2021

@author: Abdul Qayyum
"""
# Covid classification using deep learning models
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
torch.cuda.device_count()  # print 1
from collections import defaultdict
import copy
import random
import os
import shutil
from urllib.request import urlretrieve

import albumentations as A
#from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

cudnn.benchmark = True

import os 
import pandas as pd
from PIL import Image
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dataset
import torch
import albumentations as A
#from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

class DataSetCovid(Dataset):
    def __init__(self,root,transform):
        super().__init__()
        self.root=root
        self.transform=transform
        normalpathn=os.path.join(self.root,"covid")
        pathlstn=os.listdir(normalpathn)
        normalpathin=os.path.join(self.root,"normal")
        pathlstin=os.listdir(normalpathin)
        normalpathis=os.path.join(self.root,"pneumonia")
        pathlstis=os.listdir(normalpathis)
        
        self.classcov=[]
        for cl1 in pathlstn:
            pathn=os.path.join("covid",cl1)
            self.classcov.append((pathn,0))
        self.classnor=[]
        for cl2 in pathlstin:
            pathin=os.path.join("normal/",cl2)
            self.classnor.append((pathin,1))
    
        self.classpne=[]
        for cl3 in pathlstis:
            pathis=os.path.join("pneumonia/",cl3)
            self.classpne.append((pathis,2))
    
    
    
        self.fulllist=self.classcov+self.classnor+self.classpne
        #print(self.fulllist)
        
    def __getitem__(self,idx):
        
        paths,label=self.fulllist[idx]
        #print(sample)
        imagepath=os.path.join(self.root,paths)
        #print(imagepath)
        gray_img = cv2.imread(imagepath)
        #print(imagepath)
        #image=np.array(Image.open(imagepath))
        #image_g=np.array(Image.open(imagepath))
        image= gray_img
        #image=torch.from_numpy(image)
        #print(image.shape)
        
        if self.transform is not None:
            image=self.transform(image=image)["image"]
        image=image
        label=label
        
        return {"im1":image,
                "labl1":label}
    
    def __len__(self):
        return len(self.fulllist)


#pathtrain="C:\\Users\\Administrateur\\Desktop\\micca2021\\DFUC2021_trainset_2104271\\DFUC2021_train\\newdataset\\Classimages1-20210714T084847Z-001\\dfucdataset\\train"
#pathval="C:\\Users\\Administrateur\\Desktop\\micca2021\\DFUC2021_trainset_2104271\\DFUC2021_train\\newdataset\\Classimages1-20210714T084847Z-001\\dfucdataset\\val"

# Data augmentation for images
train_transforms = A.Compose([A.Resize(width=224, height=224),
                              A.RandomCrop(height=224, width=224),
                              # A.HorizontalFlip(p=0.5),
                              # A.VerticalFlip(p=0.5),
                              # A.RandomRotate90(p=0.5),
                              # A.Blur(p=0.3),
                              # A.CLAHE(p=0.3),
                              # #A.ColorJitter(p=0.3),
                              # A.CoarseDropout(max_holes=12, max_height=20, max_width=20, p=0.3),
                              # A.IAAAffine(shear=30, rotate=0, p=0.2, mode="constant"),
                              A.Normalize(mean=[0.5],
                                          std=[0.5],
                                          ),
                              #ToTensorV2(),
                              ])

val_transforms = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0.5],
                std=[0.5],
                ),
    #ToTensorV2(),
    ]
    )

pathtrain="/raid/Home/Users/aqayyum/EZProj/covid_classification_challenge/new_dataset/train/"
pathval="/raid/Home/Users/aqayyum/EZProj/covid_classification_challenge/new_dataset/validation/"

#dataset_train=DataSetCovid(pathtrain,transform=train_transforms)
#dataset_valid=DataSetCovid(pathval,transform=val_transforms) 

from torchvision import datasets,models,transforms
# Data augmentation and normalization for training
tran_transform=transforms.Compose([transforms.Resize((224,224)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485,0.456,0.406],[0.299,0.224,0.225])
                                   ])

valid_transform=transforms.Compose([transforms.Resize((224,224)),
                                   #transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485,0.456,0.406],[0.299,0.224,0.225])
                                   ])
dataset_train=datasets.ImageFolder(pathtrain,transform=tran_transform)
dataset_valid=datasets.ImageFolder(pathval,transform=valid_transform)

# dataset_train=DataSetCovid(pathtrain,transform=train_transforms)
# dataset_valid=DataSetCovid(pathval,transform=val_transforms)   


#len(dataset_train)
#for i in range(len(dataset_train)):
#    print(i)
#image,label=dataset_train[0]

from torch.utils.data import DataLoader
train_loader=DataLoader(dataset_train,batch_size=32,shuffle=True)
valid_loader=DataLoader(dataset_valid,batch_size=32,shuffle=False)
images,labels=next(iter(train_loader))
print(images.shape)
print(labels)

classes=['covid','normal','pneumonia']
my_distribution=np.array([6534,7151,4273])
class_weights = torch.from_numpy(np.divide(1, my_distribution)).float().to(device)
class_weights = class_weights / class_weights.sum()
for i, c in enumerate(classes):
  print('Weight for class %s: %f' % (c, class_weights.cpu().numpy()[i]))
loss_func = nn.CrossEntropyLoss(weight=class_weights)
#loss_func = nn.CrossEntropyLoss()

################################ training functions ###################
def train_fn(model,train_loader):
    model.train()
    counter=0
    training_run_loss=0.0
    train_running_correct=0.0
    for i, data in tqdm(enumerate(train_loader),total=int(len(dataset_train)/train_loader.batch_size)):
        counter+=1
        # extract dataset
        imge,label=data
#        imge=imge.to(device)
#        label=label.to(device)
        imge=imge.cuda()
        label=label.cuda()
        # zero_out the gradient
        optimizer.zero_grad()
        output=model(imge)
        loss=loss_func(output,label)
        training_run_loss+=loss.item()
        _,preds=torch.max(output.data,1)
        train_running_correct+=(preds==label).sum().item()
        loss.backward()
        optimizer.step()
    ###################### state computation ###################
    train_loss=training_run_loss/len(train_loader.dataset)
    train_loss_ep.append(train_loss)
    train_accuracy=100.* train_running_correct/len(train_loader.dataset)
    train_accuracy_ep.append(train_accuracy)
    print(f"Train Loss:{train_loss:.4f}, Train Acc:{train_accuracy:0.2f}")
    return train_loss_ep,train_accuracy_ep

########################## validation function ##################
def validation_fn(model,valid_loader):
  # evluation start
    print("validation start")
    
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i,data in tqdm(enumerate(valid_loader),total=int(len(dataset_valid)/valid_loader.batch_size)):
            imge,label=data
            #imge=imge.to(device)
           # label=label.to(device)
            imge=imge.cuda()
            label=label.cuda()
            output=model(imge)
            loss=loss_func(output,label)
            val_running_loss+=loss.item()
            _,pred=torch.max(output.data,1)
            val_running_correct+=(pred==label).sum().item()
        val_loss=val_running_loss/len(valid_loader.dataset)
        val_loss_ep.append(val_loss)
        val_accuracy=100.* val_running_correct/(len(valid_loader.dataset))
        val_accuracy_ep.append(val_accuracy)
        print(f"Val Loss:{val_loss:0.4f}, Val_Acc:{val_accuracy:0.2f}")
        return val_loss_ep,val_accuracy_ep
    
def evlaution_fn(model,valid_loader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images=images.float().to(device)
            labels=labels.to(device)
            model=model.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
def Prediciton_fn(model,valid_loader):
    model.eval()
    model.to(device)
    # keep track of the loss and predictions
    preds = np.zeros((len(valid_loader.dataset), 3)) # 3 classes
    labels = np.zeros((len(valid_loader.dataset)))
    for i, data in enumerate(valid_loader): 
        # sample data
        x, y = data
        # transfer data to GPU and correct format
        x = x.float().to(device)
        # feed the batch to the network and compute the outputs
        y_pred = model(x)
        # get the class probability predictions and save them for validation
        y_ = torch.softmax(y_pred, dim=1)
        b = i * valid_loader.batch_size
        preds[b: b + y_.size(0),:] = y_.detach().cpu().numpy()
        labels[b: b + y_.size(0)] = y.detach().cpu().numpy()
    return preds,labels

#ls
#
#!mkdir models
import timm
class ViTBase16(nn.Module):
    def __init__(self, model_name, n_classes, pretrained=False):

        super(ViTBase16, self).__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)
        # self.model.head = nn.Sequential(nn.Linear(self.model.head.in_features, 512),
        #                                 nn.Dropout(0.5),
        #                                 nn.ReLU(True),
        #                                 nn.Linear(512,n_classes),
        #                                 )
        #self.model.classifier = nn.Linear(self.model.classifier.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x

model_list=['vit_base_patch16_224', 
 'vit_base_patch16_224_in21k', 
 'vit_base_patch16_224_miil', 
 'vit_base_patch16_224_miil_in21k', 
 'vit_base_patch32_224', 
 'vit_base_patch32_224_in21k',  
 'vit_base_r26_s32_224', 
 'vit_base_r50_s16_224', 
 'vit_base_r50_s16_224_in21k',  
 'vit_base_resnet26d_224', 
 'vit_base_resnet50_224_in21k', 
 'vit_base_resnet50d_224']
for model_name in model_list:
    print(model_name)
    model = ViTBase16(model_name,n_classes=3, pretrained=True)
    model.cuda()
    import torch.optim as optim
    optimizer=optim.Adam(model.parameters(),lr=0.0001)
    train_loss_ep=[]
    train_accuracy_ep=[]
    val_loss_ep=[]
    val_accuracy_ep=[]
    lr = 3e-4
    log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'accu', 'val_loss', 'val_accu'])
    early_stop=10
    epochs=500
    best_acc = 0
    name=model_name
    trigger = 0
    for epoch in range(epochs):
        print('Epoch [%d/%d]' %(epoch, epochs))
        # train for one epoch
        train_loss_ep,train_accuracy_ep=train_fn(model,train_loader)
        train_loss_ep1=np.mean(train_loss_ep)
        train_accuracy_ep1=np.mean(train_accuracy_ep)
        #y_pred,labels=Prediciton_fn(model,valid_loader)

        val_loss_ep,val_accuracy_ep=validation_fn(model,valid_loader)
        val_loss_ep1=np.mean(val_loss_ep)
        val_accuracy_ep1=np.mean(val_accuracy_ep)
    
        print('loss %.4f - accu %.4f - val_loss %.4f - val_accu %.4f'%(train_loss_ep1, train_accuracy_ep1, val_loss_ep1, val_accuracy_ep1))

        tmp = pd.Series([epoch,lr,train_loss_ep1,train_accuracy_ep1,val_loss_ep1,val_accuracy_ep1], index=['epoch', 'lr', 'loss', 'accu', 'val_loss', 'val_accu'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models_t/%s/log.csv' %name, index=False)

        trigger += 1

        if val_accuracy_ep1 > best_acc:
            torch.save(model.state_dict(), 'models_t/%s/model_transformers.pth' %name)
            best_acc = val_accuracy_ep1
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not early_stop is None:
            if trigger >= early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()    
    
    
    
    
    
    
    