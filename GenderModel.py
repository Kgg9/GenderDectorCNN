# The Imports Needed For Running This Project

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.nn import functional as F
import torchvision.transforms as T
import torch.optim as optim
import os

# Implemented the Network Architecture With Forward Propagation
class ConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=(5, 5))

        self.dl1 = nn.Linear(in_features=18 * 22 * 22, out_features=180)
        self.dl2 = nn.Linear(in_features=180, out_features=90)
        self.out = nn.Linear(in_features=90, out_features=2)

    def forward(self, I):
        # First Convolutional Layer
        I = self.conv1(I)
        I = F.relu(I)
        I = F.max_pool2d(I, kernel_size=2, stride=2)

        # Second Convolutional Layer
        I = self.conv2(I)
        I = F.relu(I)
        I = F.max_pool2d(I, kernel_size=2, stride=2)

        # First Linear Layer
        I = torch.flatten(I, 1)
        I = self.dl1(I)
        I = F.relu(I)

        # Second Linear Layer
        I = self.dl2(I)
        I = F.relu(I)

        # Output Layer
        I = self.out(I)

        return I

def main():

    # Gets the RGB data from the images and converts it into a 3 dimensional array,
    # then puts it into the X_train array, with the Y_trainlabel array being the labels for the training set
    X_train = []
    Y_trainLabel = []

    def dataSet():
        PATH = "archive/Training"
        fields = ["male", "female"]

        for gender in fields:
            path = os.path.join(PATH, gender)
            for image in os.listdir(path):
                imgArray = cv2.imread(os.path.join(path, image))
                X_train.append(imgArray)
                Y_trainLabel.append(fields.index(gender))

    dataSet()



    # Custom Dataset class was made to convert the 3d array into tensors after resizing each picture into 200 x 200 pixels

    class MaleAndFemaleDataSet(Dataset):
        def __init__(self,X_train, Y_trainLabel, transform = T.Compose([T.ToPILImage(),T.Resize((100,100)),T.ToTensor()]) ):
            self.xTrain = X_train
            self.ylabel = Y_trainLabel
            self.transform = transform

        def __len__(self):
            return len(self.xTrain)

        def __getitem__(self, index):
            image = self.xTrain[index]
            label = torch.tensor(self.ylabel[index])
            X = self.transform(image)
            return (X,label)

    trainingData = MaleAndFemaleDataSet(X_train,Y_trainLabel)

    # Tells us how many predicted values were correct
    def numberofCorrect(preds,labels):
        return preds.argmax(dim=1).eq(labels).sum().item()


    # The neural network being trained
    network = ConvNetwork() # instance of the neural network

    for epoch in range(5): # number of epochs which will be ran
        trainingDataLoader = DataLoader(dataset=trainingData,batch_size=50,shuffle=True) # batches of the training data
        optimizer = optim.SGD(network.parameters(), lr=0.03) # optimizing algorithm which changes the weights of the neural network

        total_loss = 0
        total_correct = 0

        for batch in trainingDataLoader: # for loop for training the nural network on each batch
            images, labels = batch
            preds =network(images) # prediction given out by the network

            loss = F.cross_entropy(preds,labels) # calculates the loss (prediction - actual)

            optimizer.zero_grad() # manually resets the gradients since pytorch adds on the new gradients
            loss.backward() # does back-propagation and upgrades the gradients
            optimizer.step() # updates the weights for the neural network

            total_loss+=loss.item()
            total_correct+=numberofCorrect(preds, labels)

        print("epoch:",epoch,"Total Correct:", total_correct, "Total Loss", total_loss)
    torch.save(network.state_dict(), "modelM.pth")

if __name__ == "__main__":
   main()