import cv2
import numpy as np
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, Lambda
from PIL import Image
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR

base_dir = os.getcwd()
training_label_dir = os.path.join(base_dir, r"train\labels.csv")
test_label_dir = os.path.join(base_dir, r"test\labels.csv")


batch_size = 4
transform_size = 224
test_acc = []
train_acc = []

def Crop_Image(oimg):
    img = np.array(oimg.convert('L'))
    img[np.where(img >= 230)] = 255
    gray_img = 255 - img
    by_row = np.sum(gray_img, axis = 1)
    xind = np.where(by_row != 0)[0]
    xind_0 = np.where(by_row == 0)[0]
    tmp1 = xind[0]
    tmp2 = np.where(xind_0 > tmp1)[0]
    tmp3 = xind_0[tmp2[0]]
    tmp4 = np.where(xind > tmp3)[0]
    xl = xind[tmp4[0]]
    xr = xind[-1]

    by_col = np.sum(gray_img, axis = 0)
    yind = np.where(by_col != 0)
    yu = yind[0][0]
    yd = yind[0][-1]

    crop = oimg.crop((yu, xl, yd, xr))
    imgout = crop.resize((transform_size, transform_size))
    return imgout


def crop(img, color):
    if(color):
        gray_img = 255 - img[:, :, 0]
    else:
        gray_img = 255 - img

    by_row = np.sum(gray_img, axis = 1)
    xind = np.where(by_row != 0)[0]
    xind_0 = np.where(by_row == 0)[0]
    tmp1 = xind[0]
    tmp2 = np.where(xind_0 > tmp1)[0]
    tmp3 = xind_0[tmp2[0]]
    tmp4 = np.where(xind > tmp3)[0]
    xl = xind[tmp4[0]]
    xr = xind[-1]

    by_col = np.sum(gray_img, axis = 0)
    yind = np.where(by_col != 0)
    yu = yind[0][0]
    yd = yind[0][-1]
    if(color):
        crop = img[xl:xr, yu:yd, :]
    else:
        crop = img[xl:xr, yu:yd]
    imgout = cv2.resize(crop, (transform_size, transform_size))
    return imgout

def read_img(dir, contour):
    
    if(contour):
        img = cv2.imread(dir, 1)
        # img = cv2.resize(img, None, fx = ratio, fy = ratio)
        img = crop(img, True)
        green = img[:, :, 1]
        red = img[:, :, 2]
        blue = img[:, :, 0]
        # contour = (green > 100) * (red < 100) * 255
        contour = np.array((green > 100) * (red < 100) * 255, dtype = np.uint8)
        img = contour
        # print(contour.dtype)
    else:
        # img = cv2.imread(dir, 1)
        # img = cv2.resize(img, None, fx = ratio, fy = ratio)
        # img = crop(img, True)
        img = Image.open(dir)
        img = Crop_Image(img)
    return img

data_transform = Compose([
    RandomHorizontalFlip(),
    ToTensor(),
])

label_transform = Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

class CellImageDataset(Dataset):
    def __init__(self, label_dir, transform = None, target_transform = None):
        self.labels = pd.read_csv(label_dir)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.labels.iloc[idx, 0])
        img = read_img(img_path, False)
        label = self.labels.iloc[idx, 1]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

training_data = CellImageDataset(training_label_dir, transform = data_transform, target_transform = None)
train_dataloader = DataLoader(training_data, batch_size = batch_size, shuffle = True)
test_data = CellImageDataset(test_label_dir, transform = data_transform, target_transform = None)
test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = True)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print("Using {} device".format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(int(transform_size ** 2), 512), # in_feature
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2) # out_feature
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# model = NeuralNetwork().to(device)
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2, bias = False)
model.to(device)
print(model)

learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
scheduler = ExponentialLR(optimizer, gamma = 0.9)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:5>d}]")
    correct /= size
    train_acc.append(correct)

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    test_acc.append(correct)
    
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 100

if __name__ == '__main__':
    # a = 0 
    # read_img(r"D:\CellNet\ncell\01019.bmp")
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-----------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model)
        scheduler.step()
    print("DONE!")
    x = np.arange(len(test_acc))
    torch.save(model.state_dict(), os.path.join(base_dir, "model.pth"))
    print("Saved Pytorch Model State to model.pth")
    plt.plot(x, test_acc, label = 'test')
    plt.plot(x, train_acc, label = 'train')
    plt.show()
    