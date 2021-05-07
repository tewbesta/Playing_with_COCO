import argparse
from PIL import Image
import glob
import torch
from torchvision import transforms as tvt
parser = argparse.ArgumentParser(description ='HW02 Task2')
parser.add_argument('--imagenet_root', type =str ,required= True)
parser.add_argument('--class_list', nargs ='*',type =str,required=True)
args, args_other=parser.parse_known_args()
from torch.utils.data import DataLoader,Dataset
import numpy as np
from collections import defaultdict
class your_dataset_class(Dataset):
    def __init__(self,x,y,transformations):
        self.class_list = x
        self.transformations=transformations
        self.imagenet_root = y
        self.cat_images=glob.glob(self.imagenet_root+self.class_list[0]+"/*.jpg")
        self.dog_images = glob.glob(self.imagenet_root + self.class_list[1] + "/*.jpg")
        self.cat_labels = torch.tensor([0, 1])
        self.dog_labels = torch.tensor([1, 0])
        dog_dict = {}
        cat_dict = {}
        for i in range(len(self.dog_images)):
            dog_images = Image.open(self.dog_images[i])
            dog_images = self.transformations(dog_images)
            dog_dict[dog_images] = self.dog_labels
            dogkey = list(dog_dict.keys())
            dogval = list(dog_dict.values())
        for i in range(len(self.cat_images)):
            cat_images = Image.open(self.cat_images[i])
            cat_images = self.transformations(cat_images)
            cat_dict[cat_images] = self.cat_labels
            catkey = list(cat_dict.keys())
            catval = list(cat_dict.values())
        images1 = {**dog_dict, **cat_dict}
        self.imageslist = list(images1.keys())
        self.labels = list(images1.values())
    def __getitem__(self,index):
        #dataset = datasets.ImageFolder(self.images, transform=self.transformations)

        return self.imageslist[index],self.labels[index]
    def __len__(self):
        self.len = len(self.imageslist)
        #self.dog_length = len(self.dog_images)
        return self.len

transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset=your_dataset_class(args.class_list, args.imagenet_root+"Train/",transform)
total=len(train_dataset)
print("the length is", total)
train_data_loader = torch.utils.data.DataLoader(dataset =train_dataset ,batch_size =10 ,shuffle =True ,num_workers =0)
val_dataset = your_dataset_class (args.class_list,args.imagenet_root+"Val/", transform)
val_data_loader = torch.utils.data.DataLoader ( dataset=val_dataset ,batch_size =10 ,shuffle =True ,num_workers =0)
totalval=len(val_dataset)
dtype = torch.float64
device = torch.device ("cuda :0" if torch.cuda.is_available()else "cpu")
epochs =80# feel free to adjust this parameter
D_in , H1 , H2 , D_out = 3*64*64 , 1000 , 256 , 2
w1 = torch.randn (D_in,H1,device =device,dtype = dtype )
w2 = torch.randn ( H1,H2,device =device,dtype = dtype )
w3 = torch.randn ( H2,D_out,device =device,dtype = dtype )
learning_rate = 1e-9
print("working with cuda", torch.cuda.is_available())
for t in range(epochs):
    epoch_loss = 0
    count=0
    for i, data in enumerate(train_data_loader):
        inputs, labels= data
        inputs = torch.FloatTensor(inputs).to(device)
        labels = (torch.as_tensor(np.array(labels))).to(device)
        x = inputs.view(inputs.size(0), -1)
        w1= w1.type(torch.FloatTensor)
        w2 = w2.type(torch.FloatTensor)
        w3 = w3.type(torch.FloatTensor)
        h1 = x.mm(w1)
        h1_relu = h1.clamp(min=0)
        h2 = h1_relu.mm(w2)
        h2_relu = h2.clamp(min=0)
        y_pred = h2_relu.mm(w3)
        loss = (y_pred-labels).pow(2).sum().item()
        y_error = y_pred-labels
        h2_error = 2.0 * y_error.mm(w3.t())
        h2_error[h2 < 0] = 0
        h1_error = 2.0 * h2_error.mm(w2.t())
        h1_error[h1<0] = 0
        grad_w1 = x.t().mm(2 * h1_error)
        grad_w2 = h1_relu.t().mm(2 * h2_error)
        grad_w3 = h2_relu.t().mm(2 * y_error)
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        w3 -= learning_rate * grad_w3
        y_pred = torch.argmax(y_pred, dim=1)
        labels = torch.argmax(labels, dim=1)
        for i in range(len(y_pred)):
            if ((y_pred[i] == labels[i])):
                count = count + 1
        print("count", count)
        epoch_loss=(loss+epoch_loss)
    print('Epoch %d:\t %0.4f' % (t, epoch_loss), file=open("output.txt", "a"))
torch.save({'w1 ':w1,'w2 ':w2,'w3 ':w3}, './wts.pkl')
print("Val")
for t in range(epochs):
    length=0
    epoch_loss = 0
    count=0
    for i, data in enumerate(val_data_loader):
        inputs, labels= data
        inputs = torch.FloatTensor(inputs).to(device)
        labels = (torch.as_tensor(np.array(labels))).to(device)
        x = inputs.view(inputs.size(0), -1)
        w1= w1.type(torch.FloatTensor)
        w2 = w2.type(torch.FloatTensor)
        w3 = w3.type(torch.FloatTensor)
        h1 = x.mm(w1)
        h1_relu = h1.clamp(min=0)
        h2 = h1_relu.mm(w2)
        h2_relu = h2.clamp(min=0)
        y_pred = h2_relu.mm(w3)
        loss = (y_pred - labels).pow(2).sum().item()
        y_error = y_pred - labels
        h2_error = 2.0 * y_error.mm(w3.t())
        h2_error[h2 < 0] = 0
        h1_error = 2.0 * h2_error.mm(w2.t())
        h1_error[h1 < 0] = 0
        grad_w1 = x.t().mm(2 * h1_error)
        grad_w2 = h1_relu.t().mm(2 * h2_error)
        grad_w3 = h2_relu.t().mm(2 * y_error)
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        w3 -= learning_rate * grad_w3
        y_pred = torch.argmax(y_pred, dim=1)
        labels = torch.argmax(labels, dim=1)
        print(y_pred.shape, labels.shape)
        for i in range(len(y_pred)):
            if ((y_pred[i] == labels[i])):
                count = count + 1
        print("count", count)
    print('Val accuracy percentage is for epoch %d:\t %0.4f'% (t, ((count/totalval)*100)), file=open("output.txt", "a"))
torch.save({'w1 ':w1,'w2 ':w2,'w3 ':w3}, './val.pkl')

