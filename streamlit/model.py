import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
from torch.autograd import Variable

# CNN Network
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet,self).__init__()
        
        # Input shape= (256,3,150,150)
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        # Shape = (256,12,150,150)
        self.bn1=nn.BatchNorm2d(num_features=12)
        self.relu1=nn.ReLU()
        
        # Reduce the image size be factor 2
        # Shape = (256,12,75,75)
        self.pool=nn.MaxPool2d(kernel_size=2)
        
        # Shape = (256,20,75,75)
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        
        # Shape = (256,32,75,75)
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(num_features=32)
        self.relu3=nn.ReLU()
        
        self.fc=nn.Linear(in_features=75 * 75 * 32, out_features = num_classes)
        
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
            
            
        #Above output will be in matrix form, with shape (256,32,75,75) 
        output=output.view(-1,32*75*75)
        output=self.fc(output)
            
        return output
    
    def transformer(img_width, img_height):
        transformer = transforms.Compose([
            transforms.Resize((img_width, img_height)),
            transforms.ToTensor(), #0-255 to 0-1, numpy to tensors
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # 0-1 to [-1, 1], formula (x-mean)/std
        ])

        return transformer


# Predic Model
class Model:
    def __init__(self, model, classes):
        self.model = model
        self.classes = classes

    def prediction(self, img_path, transformer):
        image = Image.open(img_path)
        image_tensor=transformer(image).float()
        image_tensor=image_tensor.unsqueeze_(0)
            
        input = Variable(image_tensor)

        output = self.model(input)
        
        index=output.data.numpy().argmax()
        
        pred = self.classes[index]
        
        return pred