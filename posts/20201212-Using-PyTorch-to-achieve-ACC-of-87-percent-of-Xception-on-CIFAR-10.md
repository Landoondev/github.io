# Xception 复现

如下为完整的 Jupyter Notebook。

```python
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import glob
import PIL
from PIL import Image
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler
import random
```


```python
batch_size = 64
validation_ratio = 0.1
random_seed = 10
```


```python
transform_train = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomCrop(128, padding=32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

transform_validation = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])


transform_test = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

validset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_validation)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                          shuffle=True, num_workers=0)

num_train = len(trainset)
indices = list(range(num_train))
split = int(np.floor(validation_ratio * num_train))

np.random.seed(random_seed)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, sampler=train_sampler, num_workers=0
)

valid_loader = torch.utils.data.DataLoader(
    validset, batch_size=batch_size, sampler=valid_sampler, num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=0
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

initial_lr = 0.045
```

    Files already downloaded and verified
    Files already downloaded and verified
    Files already downloaded and verified



```python
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
```


```python
class Xception(nn.Module):
    def __init__(self, input_channel, num_classes=10):
        super(Xception, self).__init__()
        
        # Entry Flow
        self.entry_flow_1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.entry_flow_2 = nn.Sequential(
            depthwise_separable_conv(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            depthwise_separable_conv(128, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.entry_flow_2_residual = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)
        
        self.entry_flow_3 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(128, 256, 3, 1),
            nn.BatchNorm2d(256),
            
            nn.ReLU(True),
            depthwise_separable_conv(256, 256, 3, 1),
            nn.BatchNorm2d(256),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.entry_flow_3_residual = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)
        
        self.entry_flow_4 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(256, 728, 3, 1),
            nn.BatchNorm2d(728),
            
            nn.ReLU(True),
            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.entry_flow_4_residual = nn.Conv2d(256, 728, kernel_size=1, stride=2, padding=0)
        
        # Middle Flow
        self.middle_flow = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728),
            
            nn.ReLU(True),
            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728),
            
            nn.ReLU(True),
            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728)
        )
        
        # Exit Flow
        self.exit_flow_1 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728),
            
            nn.ReLU(True),
            depthwise_separable_conv(728, 1024, 3, 1),
            nn.BatchNorm2d(1024),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.exit_flow_1_residual = nn.Conv2d(728, 1024, kernel_size=1, stride=2, padding=0)
        self.exit_flow_2 = nn.Sequential(
            depthwise_separable_conv(1024, 1536, 3, 1),
            nn.BatchNorm2d(1536),
            nn.ReLU(True),
            
            depthwise_separable_conv(1536, 2048, 3, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(True)
        )
        
        self.linear = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        entry_out1 = self.entry_flow_1(x)
        entry_out2 = self.entry_flow_2(entry_out1) + self.entry_flow_2_residual(entry_out1)
        entry_out3 = self.entry_flow_3(entry_out2) + self.entry_flow_3_residual(entry_out2)
        entry_out = self.entry_flow_4(entry_out3) + self.entry_flow_4_residual(entry_out3)
        
        middle_out = self.middle_flow(entry_out) + entry_out
        
        for i in range(7):
          middle_out = self.middle_flow(middle_out) + middle_out

        exit_out1 = self.exit_flow_1(middle_out) + self.exit_flow_1_residual(middle_out)
        exit_out2 = self.exit_flow_2(exit_out1)

        exit_avg_pool = F.adaptive_avg_pool2d(exit_out2, (1, 1))                
        exit_avg_pool_flat = exit_avg_pool.view(exit_avg_pool.size(0), -1)

        output = self.linear(exit_avg_pool_flat)
        
        return output
```


```python
net = Xception(3, 10) #ResNet-18
```


```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```

    cuda:0



```python
net.to(device)
```




    Xception(
      (entry_flow_1): Sequential(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
      )
      (entry_flow_2): Sequential(
        (0): depthwise_separable_conv(
          (depthwise): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (pointwise): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): depthwise_separable_conv(
          (depthwise): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (pointwise): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
      (entry_flow_2_residual): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2))
      (entry_flow_3): Sequential(
        (0): ReLU(inplace=True)
        (1): depthwise_separable_conv(
          (depthwise): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (pointwise): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): depthwise_separable_conv(
          (depthwise): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (pointwise): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
      (entry_flow_3_residual): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2))
      (entry_flow_4): Sequential(
        (0): ReLU(inplace=True)
        (1): depthwise_separable_conv(
          (depthwise): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (pointwise): Conv2d(256, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): depthwise_separable_conv(
          (depthwise): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
      (entry_flow_4_residual): Conv2d(256, 728, kernel_size=(1, 1), stride=(2, 2))
      (middle_flow): Sequential(
        (0): ReLU(inplace=True)
        (1): depthwise_separable_conv(
          (depthwise): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): depthwise_separable_conv(
          (depthwise): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): depthwise_separable_conv(
          (depthwise): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (exit_flow_1): Sequential(
        (0): ReLU(inplace=True)
        (1): depthwise_separable_conv(
          (depthwise): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): depthwise_separable_conv(
          (depthwise): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
      (exit_flow_1_residual): Conv2d(728, 1024, kernel_size=(1, 1), stride=(2, 2))
      (exit_flow_2): Sequential(
        (0): depthwise_separable_conv(
          (depthwise): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024, bias=False)
          (pointwise): Conv2d(1024, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (1): BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): depthwise_separable_conv(
          (depthwise): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536, bias=False)
          (pointwise): Conv2d(1536, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (4): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
      )
      (linear): Linear(in_features=2048, out_features=10, bias=True)
    )




```python
import time
start_time = time.asctime(time.localtime(time.time()))
print ("Train Start at: ",start_time)
```

    Train Start at:  Fri Dec 11 08:35:43 2020

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9)

for epoch in range(200):
    if epoch == 0:
        lr = initial_lr
    elif epoch % 2 == 0 and epoch != 0:
        lr *= 0.94
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        show_period = 250
        if i % show_period == show_period-1:    # print every "show_period" mini-batches
            print('[%d, %5d] loss: %.7f' %
                  (epoch + 1, i + 1, running_loss / show_period))
            running_loss = 0.0
        
        
    #validation part
    correct = 0
    total = 0
    for i, data in enumerate(valid_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print('[%d epoch] Accuracy of the network on the validation images: %d %%' % 
          (epoch, 100 * correct / total)
         )

print('Finished Training')
```

    [1,   250] loss: 2.1078744
    [1,   500] loss: 1.7013918
    [0 epoch] Accuracy of the network on the validation images: 42 %
    [2,   250] loss: 1.3789000
    [2,   500] loss: 1.2187569
    [1 epoch] Accuracy of the network on the validation images: 56 %
    [3,   250] loss: 1.0353949
    [3,   500] loss: 0.9460982
    [2 epoch] Accuracy of the network on the validation images: 62 %
    [4,   250] loss: 0.8064175
    [4,   500] loss: 0.7470114
    [3 epoch] Accuracy of the network on the validation images: 67 %
    [5,   250] loss: 0.6502923
    [5,   500] loss: 0.6354827
    [4 epoch] Accuracy of the network on the validation images: 72 %
    [6,   250] loss: 0.5987351
    [6,   500] loss: 0.5660082
    [5 epoch] Accuracy of the network on the validation images: 74 %
    [7,   250] loss: 0.4956860
    [7,   500] loss: 0.4744828
    [6 epoch] Accuracy of the network on the validation images: 74 %
    [8,   250] loss: 0.5037934
    [8,   500] loss: 0.4528574
    [7 epoch] Accuracy of the network on the validation images: 76 %
    [9,   250] loss: 0.3893477
    [9,   500] loss: 0.3898181
    [8 epoch] Accuracy of the network on the validation images: 76 %
    [10,   250] loss: 0.3783518
    [10,   500] loss: 0.3650177
    [9 epoch] Accuracy of the network on the validation images: 77 %
    [11,   250] loss: 0.3137898
    [11,   500] loss: 0.3314004
    [10 epoch] Accuracy of the network on the validation images: 78 %
    [12,   250] loss: 0.4802624
    [12,   500] loss: 0.3640091
    [11 epoch] Accuracy of the network on the validation images: 77 %
    [13,   250] loss: 0.3035468
    [13,   500] loss: 0.3064248
    [12 epoch] Accuracy of the network on the validation images: 79 %
    [14,   250] loss: 0.2797937
    [14,   500] loss: 0.2865547
    [13 epoch] Accuracy of the network on the validation images: 79 %
    [15,   250] loss: 0.2390145
    [15,   500] loss: 0.2555036
    [14 epoch] Accuracy of the network on the validation images: 80 %
    [16,   250] loss: 0.2436917
    [16,   500] loss: 0.2438789
    [15 epoch] Accuracy of the network on the validation images: 78 %
    [17,   250] loss: 0.2153499
    [17,   500] loss: 0.2273009
    [16 epoch] Accuracy of the network on the validation images: 80 %
    [18,   250] loss: 0.2005612
    [18,   500] loss: 0.2011595
    [17 epoch] Accuracy of the network on the validation images: 81 %
    [19,   250] loss: 0.1808655
    [19,   500] loss: 0.1849401
    [18 epoch] Accuracy of the network on the validation images: 80 %
    [20,   250] loss: 0.1749664
    [20,   500] loss: 0.1758422
    [19 epoch] Accuracy of the network on the validation images: 81 %
    [21,   250] loss: 0.1600653
    [21,   500] loss: 0.1541128
    [20 epoch] Accuracy of the network on the validation images: 81 %
    [22,   250] loss: 0.1546641
    [22,   500] loss: 0.1579930
    [21 epoch] Accuracy of the network on the validation images: 81 %
    [23,   250] loss: 0.1358909
    [23,   500] loss: 0.1414411
    [22 epoch] Accuracy of the network on the validation images: 82 %
    [24,   250] loss: 0.1270603
    [24,   500] loss: 0.1295347
    [23 epoch] Accuracy of the network on the validation images: 82 %
    [25,   250] loss: 0.1158701
    [25,   500] loss: 0.1236949
    [24 epoch] Accuracy of the network on the validation images: 83 %
    [26,   250] loss: 0.1141707
    [26,   500] loss: 0.1159905
    [25 epoch] Accuracy of the network on the validation images: 82 %
    [27,   250] loss: 0.1007193
    [27,   500] loss: 0.1026152
    [26 epoch] Accuracy of the network on the validation images: 82 %
    [28,   250] loss: 0.1053027
    [28,   500] loss: 0.1015101
    [27 epoch] Accuracy of the network on the validation images: 82 %
    [29,   250] loss: 0.0849384
    [29,   500] loss: 0.0954018
    [28 epoch] Accuracy of the network on the validation images: 82 %
    [30,   250] loss: 0.0882779
    [30,   500] loss: 0.0844315
    [29 epoch] Accuracy of the network on the validation images: 83 %
    [31,   250] loss: 0.0755459
    [31,   500] loss: 0.0748325
    [30 epoch] Accuracy of the network on the validation images: 82 %
    [32,   250] loss: 0.0752830
    [32,   500] loss: 0.0764719
    [31 epoch] Accuracy of the network on the validation images: 83 %
    [33,   250] loss: 0.0631574
    [33,   500] loss: 0.0670219
    [32 epoch] Accuracy of the network on the validation images: 83 %
    [34,   250] loss: 0.0617852
    [34,   500] loss: 0.0604056
    [33 epoch] Accuracy of the network on the validation images: 83 %
    [35,   250] loss: 0.0569031
    [35,   500] loss: 0.0589615
    [34 epoch] Accuracy of the network on the validation images: 83 %
    [36,   250] loss: 0.0737599
    [36,   500] loss: 0.0597042
    [35 epoch] Accuracy of the network on the validation images: 83 %
    [37,   250] loss: 0.0586099
    [37,   500] loss: 0.0551216
    [36 epoch] Accuracy of the network on the validation images: 83 %
    [38,   250] loss: 0.0526482
    [38,   500] loss: 0.0513871
    [37 epoch] Accuracy of the network on the validation images: 83 %
    [39,   250] loss: 0.0422603
    [39,   500] loss: 0.0478268
    [38 epoch] Accuracy of the network on the validation images: 82 %
    [40,   250] loss: 0.0431830
    [40,   500] loss: 0.0446909
    [39 epoch] Accuracy of the network on the validation images: 83 %
    [41,   250] loss: 0.0364558
    [41,   500] loss: 0.0384867
    [40 epoch] Accuracy of the network on the validation images: 84 %
    [42,   250] loss: 0.0416089
    [42,   500] loss: 0.0419357
    [41 epoch] Accuracy of the network on the validation images: 83 %
    [43,   250] loss: 0.0371038
    [43,   500] loss: 0.0371064
    [42 epoch] Accuracy of the network on the validation images: 84 %
    [44,   250] loss: 0.0331247
    [44,   500] loss: 0.0352282
    [43 epoch] Accuracy of the network on the validation images: 83 %
    [45,   250] loss: 0.0281234
    [45,   500] loss: 0.0331068
    [44 epoch] Accuracy of the network on the validation images: 84 %
    [46,   250] loss: 0.0387601
    [46,   500] loss: 0.0318495
    [45 epoch] Accuracy of the network on the validation images: 84 %
    [47,   250] loss: 0.0287145
    [47,   500] loss: 0.0295377
    [46 epoch] Accuracy of the network on the validation images: 83 %
    [48,   250] loss: 0.0369417
    [48,   500] loss: 0.0265885
    [47 epoch] Accuracy of the network on the validation images: 83 %
    [49,   250] loss: 0.0251548
    [49,   500] loss: 0.0263732
    [48 epoch] Accuracy of the network on the validation images: 84 %
    [50,   250] loss: 0.0434540
    [50,   500] loss: 0.0312673
    [49 epoch] Accuracy of the network on the validation images: 83 %
    [51,   250] loss: 0.0225699
    [51,   500] loss: 0.0266013
    [50 epoch] Accuracy of the network on the validation images: 84 %
    [52,   250] loss: 0.0220982
    [52,   500] loss: 0.0241781
    [51 epoch] Accuracy of the network on the validation images: 84 %
    [53,   250] loss: 0.0184493
    [53,   500] loss: 0.0201668
    [52 epoch] Accuracy of the network on the validation images: 85 %
    [54,   250] loss: 0.0218614
    [54,   500] loss: 0.0212918
    [53 epoch] Accuracy of the network on the validation images: 84 %
    [55,   250] loss: 0.0181113
    [55,   500] loss: 0.0210739
    [54 epoch] Accuracy of the network on the validation images: 84 %
    [56,   250] loss: 0.0246859
    [56,   500] loss: 0.0180740
    [55 epoch] Accuracy of the network on the validation images: 84 %
    [57,   250] loss: 0.0174577
    [57,   500] loss: 0.0167248
    [56 epoch] Accuracy of the network on the validation images: 84 %
    [58,   250] loss: 0.0227070
    [58,   500] loss: 0.0154899
    [57 epoch] Accuracy of the network on the validation images: 84 %
    [59,   250] loss: 0.0145918
    [59,   500] loss: 0.0163994
    [58 epoch] Accuracy of the network on the validation images: 84 %
    [60,   250] loss: 0.0179376
    [60,   500] loss: 0.0164890
    [59 epoch] Accuracy of the network on the validation images: 84 %
    [61,   250] loss: 0.0120257
    [61,   500] loss: 0.0136571
    [60 epoch] Accuracy of the network on the validation images: 85 %
    [62,   250] loss: 0.0164738
    [62,   500] loss: 0.0130985
    [61 epoch] Accuracy of the network on the validation images: 85 %
    [63,   250] loss: 0.0122985
    [63,   500] loss: 0.0102551
    [62 epoch] Accuracy of the network on the validation images: 84 %
    [64,   250] loss: 0.0107730
    [64,   500] loss: 0.0142992
    [63 epoch] Accuracy of the network on the validation images: 85 %
    [65,   250] loss: 0.0121013
    [65,   500] loss: 0.0125776
    [64 epoch] Accuracy of the network on the validation images: 84 %
    [66,   250] loss: 0.0318173
    [66,   500] loss: 0.0177379
    [65 epoch] Accuracy of the network on the validation images: 84 %
    [67,   250] loss: 0.0136800
    [67,   500] loss: 0.0133418
    [66 epoch] Accuracy of the network on the validation images: 84 %
    [68,   250] loss: 0.0126889
    [68,   500] loss: 0.0129200
    [67 epoch] Accuracy of the network on the validation images: 85 %
    [69,   250] loss: 0.0112317
    [69,   500] loss: 0.0116105
    [68 epoch] Accuracy of the network on the validation images: 85 %
    [70,   250] loss: 0.0119080
    [70,   500] loss: 0.0097267
    [69 epoch] Accuracy of the network on the validation images: 85 %
    [71,   250] loss: 0.0115536
    [71,   500] loss: 0.0094735
    [70 epoch] Accuracy of the network on the validation images: 86 %
    [72,   250] loss: 0.0105987
    [72,   500] loss: 0.0119858
    [71 epoch] Accuracy of the network on the validation images: 85 %
    [73,   250] loss: 0.0121387
    [73,   500] loss: 0.0092085
    [72 epoch] Accuracy of the network on the validation images: 86 %
    [74,   250] loss: 0.0070642
    [74,   500] loss: 0.0081908
    [73 epoch] Accuracy of the network on the validation images: 86 %
    [75,   250] loss: 0.0099117
    [75,   500] loss: 0.0085511
    [74 epoch] Accuracy of the network on the validation images: 86 %
    [76,   250] loss: 0.0076440
    [76,   500] loss: 0.0074626
    [75 epoch] Accuracy of the network on the validation images: 86 %
    [77,   250] loss: 0.0062244
    [77,   500] loss: 0.0096577
    [76 epoch] Accuracy of the network on the validation images: 86 %
    [78,   250] loss: 0.0081819
    [78,   500] loss: 0.0065762
    [77 epoch] Accuracy of the network on the validation images: 86 %
    [79,   250] loss: 0.0055625
    [79,   500] loss: 0.0073242
    [78 epoch] Accuracy of the network on the validation images: 86 %
    [80,   250] loss: 0.0081289
    [80,   500] loss: 0.0065801
    [79 epoch] Accuracy of the network on the validation images: 86 %
    [81,   250] loss: 0.0080191
    [81,   500] loss: 0.0067672
    [80 epoch] Accuracy of the network on the validation images: 86 %
    [82,   250] loss: 0.0091357
    [82,   500] loss: 0.0081149
    [81 epoch] Accuracy of the network on the validation images: 86 %
    [83,   250] loss: 0.0083918
    [83,   500] loss: 0.0087531
    [82 epoch] Accuracy of the network on the validation images: 86 %
    [84,   250] loss: 0.0056003
    [84,   500] loss: 0.0057701
    [83 epoch] Accuracy of the network on the validation images: 86 %
    [85,   250] loss: 0.0061889
    [85,   500] loss: 0.0051294
    [84 epoch] Accuracy of the network on the validation images: 86 %
    [86,   250] loss: 0.0063507
    [86,   500] loss: 0.0059396
    [85 epoch] Accuracy of the network on the validation images: 85 %
    [87,   250] loss: 0.0060617
    [87,   500] loss: 0.0079281
    [86 epoch] Accuracy of the network on the validation images: 85 %
    [88,   250] loss: 0.0060172
    [88,   500] loss: 0.0050122
    [87 epoch] Accuracy of the network on the validation images: 85 %
    [89,   250] loss: 0.0036895
    [89,   500] loss: 0.0054657
    [88 epoch] Accuracy of the network on the validation images: 85 %
    [90,   250] loss: 0.0064730
    [90,   500] loss: 0.0071091
    [89 epoch] Accuracy of the network on the validation images: 85 %
    [91,   250] loss: 0.0056076
    [91,   500] loss: 0.0047377
    [90 epoch] Accuracy of the network on the validation images: 85 %
    [92,   250] loss: 0.0052442
    [92,   500] loss: 0.0053140
    [91 epoch] Accuracy of the network on the validation images: 85 %
    [93,   250] loss: 0.0052133
    [93,   500] loss: 0.0055943
    [92 epoch] Accuracy of the network on the validation images: 85 %
    [94,   250] loss: 0.0061436
    [94,   500] loss: 0.0037984
    [93 epoch] Accuracy of the network on the validation images: 86 %
    [95,   250] loss: 0.0058168
    [95,   500] loss: 0.0047448
    [94 epoch] Accuracy of the network on the validation images: 86 %
    [96,   250] loss: 0.0044690
    [96,   500] loss: 0.0049961
    [95 epoch] Accuracy of the network on the validation images: 85 %
    [97,   250] loss: 0.0047672
    [97,   500] loss: 0.0040327
    [96 epoch] Accuracy of the network on the validation images: 86 %
    [98,   250] loss: 0.0055643
    [98,   500] loss: 0.0037597
    [97 epoch] Accuracy of the network on the validation images: 86 %
    [99,   250] loss: 0.0034475
    [99,   500] loss: 0.0045113
    [98 epoch] Accuracy of the network on the validation images: 85 %
    [100,   250] loss: 0.0049196
    [100,   500] loss: 0.0043493
    [99 epoch] Accuracy of the network on the validation images: 85 %
    [101,   250] loss: 0.0052216
    [101,   500] loss: 0.0046963
    [100 epoch] Accuracy of the network on the validation images: 85 %
    [102,   250] loss: 0.0044481
    [102,   500] loss: 0.0040119
    [101 epoch] Accuracy of the network on the validation images: 85 %
    [103,   250] loss: 0.0037515
    [103,   500] loss: 0.0044648
    [102 epoch] Accuracy of the network on the validation images: 86 %
    [104,   250] loss: 0.0044892
    [104,   500] loss: 0.0041154
    [103 epoch] Accuracy of the network on the validation images: 86 %
    [105,   250] loss: 0.0034194
    [105,   500] loss: 0.0041636
    [104 epoch] Accuracy of the network on the validation images: 86 %
    [106,   250] loss: 0.0035675
    [106,   500] loss: 0.0046522
    [105 epoch] Accuracy of the network on the validation images: 85 %
    [107,   250] loss: 0.0034231
    [107,   500] loss: 0.0042649
    [106 epoch] Accuracy of the network on the validation images: 86 %
    [108,   250] loss: 0.0040298
    [108,   500] loss: 0.0039121
    [107 epoch] Accuracy of the network on the validation images: 86 %
    [109,   250] loss: 0.0037955
    [109,   500] loss: 0.0031334
    [108 epoch] Accuracy of the network on the validation images: 85 %
    [110,   250] loss: 0.0043019
    [110,   500] loss: 0.0039885
    [109 epoch] Accuracy of the network on the validation images: 85 %
    [111,   250] loss: 0.0037174
    [111,   500] loss: 0.0031499
    [110 epoch] Accuracy of the network on the validation images: 85 %
    [112,   250] loss: 0.0044050
    [112,   500] loss: 0.0041636
    [111 epoch] Accuracy of the network on the validation images: 86 %
    [113,   250] loss: 0.0032088
    [113,   500] loss: 0.0038616
    [112 epoch] Accuracy of the network on the validation images: 85 %
    [114,   250] loss: 0.0037744
    [114,   500] loss: 0.0036387
    [113 epoch] Accuracy of the network on the validation images: 86 %
    [115,   250] loss: 0.0032333
    [115,   500] loss: 0.0027883
    [114 epoch] Accuracy of the network on the validation images: 86 %
    [116,   250] loss: 0.0034194
    [116,   500] loss: 0.0026087
    [115 epoch] Accuracy of the network on the validation images: 86 %
    [117,   250] loss: 0.0021975
    [117,   500] loss: 0.0028611
    [116 epoch] Accuracy of the network on the validation images: 86 %
    [118,   250] loss: 0.0029720
    [118,   500] loss: 0.0044415
    [117 epoch] Accuracy of the network on the validation images: 86 %
    [119,   250] loss: 0.0038551
    [119,   500] loss: 0.0027357
    [118 epoch] Accuracy of the network on the validation images: 85 %
    [120,   250] loss: 0.0026606
    [120,   500] loss: 0.0032779
    [119 epoch] Accuracy of the network on the validation images: 86 %
    [121,   250] loss: 0.0026804
    [121,   500] loss: 0.0036855
    [120 epoch] Accuracy of the network on the validation images: 85 %
    [122,   250] loss: 0.0026897
    [122,   500] loss: 0.0032359
    [121 epoch] Accuracy of the network on the validation images: 86 %
    [123,   250] loss: 0.0036553
    [123,   500] loss: 0.0041091
    [122 epoch] Accuracy of the network on the validation images: 86 %
    [124,   250] loss: 0.0032643
    [124,   500] loss: 0.0034761
    [123 epoch] Accuracy of the network on the validation images: 85 %
    [125,   250] loss: 0.0026608
    [125,   500] loss: 0.0040761
    [124 epoch] Accuracy of the network on the validation images: 86 %
    [126,   250] loss: 0.0026380
    [126,   500] loss: 0.0021082
    [125 epoch] Accuracy of the network on the validation images: 86 %
    [127,   250] loss: 0.0032954
    [127,   500] loss: 0.0037990
    [126 epoch] Accuracy of the network on the validation images: 86 %
    [128,   250] loss: 0.0028140
    [128,   500] loss: 0.0030236
    [127 epoch] Accuracy of the network on the validation images: 85 %
    [129,   250] loss: 0.0028834
    [129,   500] loss: 0.0031183
    [128 epoch] Accuracy of the network on the validation images: 86 %
    [130,   250] loss: 0.0039143
    [130,   500] loss: 0.0034973
    [129 epoch] Accuracy of the network on the validation images: 85 %
    [131,   250] loss: 0.0016813
    [131,   500] loss: 0.0029175
    [130 epoch] Accuracy of the network on the validation images: 86 %
    [132,   250] loss: 0.0027880
    [132,   500] loss: 0.0025465
    [131 epoch] Accuracy of the network on the validation images: 86 %
    [133,   250] loss: 0.0039116
    [133,   500] loss: 0.0034565
    [132 epoch] Accuracy of the network on the validation images: 86 %
    [134,   250] loss: 0.0030665
    [134,   500] loss: 0.0030498
    [133 epoch] Accuracy of the network on the validation images: 86 %
    [135,   250] loss: 0.0030490
    [135,   500] loss: 0.0020285
    [134 epoch] Accuracy of the network on the validation images: 86 %
    [136,   250] loss: 0.0040877
    [136,   500] loss: 0.0036548
    [135 epoch] Accuracy of the network on the validation images: 86 %
    [137,   250] loss: 0.0031683
    [137,   500] loss: 0.0027891
    [136 epoch] Accuracy of the network on the validation images: 86 %
    [138,   250] loss: 0.0026111
    [138,   500] loss: 0.0037556
    [137 epoch] Accuracy of the network on the validation images: 86 %
    [139,   250] loss: 0.0028837
    [139,   500] loss: 0.0034781
    [138 epoch] Accuracy of the network on the validation images: 86 %
    [140,   250] loss: 0.0025157
    [140,   500] loss: 0.0020306
    [139 epoch] Accuracy of the network on the validation images: 86 %
    [141,   250] loss: 0.0022864
    [141,   500] loss: 0.0033019
    [140 epoch] Accuracy of the network on the validation images: 85 %
    [142,   250] loss: 0.0023374
    [142,   500] loss: 0.0021323
    [141 epoch] Accuracy of the network on the validation images: 86 %
    [143,   250] loss: 0.0029830
    [143,   500] loss: 0.0026743
    [142 epoch] Accuracy of the network on the validation images: 86 %
    [144,   250] loss: 0.0024923
    [144,   500] loss: 0.0026146
    [143 epoch] Accuracy of the network on the validation images: 86 %
    [145,   250] loss: 0.0017612
    [145,   500] loss: 0.0031088
    [144 epoch] Accuracy of the network on the validation images: 86 %
    [146,   250] loss: 0.0025228
    [146,   500] loss: 0.0031492
    [145 epoch] Accuracy of the network on the validation images: 86 %
    [147,   250] loss: 0.0023881
    [147,   500] loss: 0.0034650
    [146 epoch] Accuracy of the network on the validation images: 86 %
    [148,   250] loss: 0.0022417
    [148,   500] loss: 0.0021370
    [147 epoch] Accuracy of the network on the validation images: 86 %
    [149,   250] loss: 0.0025878
    [149,   500] loss: 0.0032328
    [148 epoch] Accuracy of the network on the validation images: 85 %
    [150,   250] loss: 0.0024920
    [150,   500] loss: 0.0033428
    [149 epoch] Accuracy of the network on the validation images: 85 %
    [151,   250] loss: 0.0025919
    [151,   500] loss: 0.0027105
    [150 epoch] Accuracy of the network on the validation images: 86 %
    [152,   250] loss: 0.0027436
    [152,   500] loss: 0.0035176
    [151 epoch] Accuracy of the network on the validation images: 86 %
    [153,   250] loss: 0.0020813
    [153,   500] loss: 0.0022240
    [152 epoch] Accuracy of the network on the validation images: 86 %
    [154,   250] loss: 0.0022741
    [154,   500] loss: 0.0024855
    [153 epoch] Accuracy of the network on the validation images: 86 %
    [155,   250] loss: 0.0020212
    [155,   500] loss: 0.0024496
    [154 epoch] Accuracy of the network on the validation images: 86 %
    [156,   250] loss: 0.0028731
    [156,   500] loss: 0.0034748
    [155 epoch] Accuracy of the network on the validation images: 86 %
    [157,   250] loss: 0.0027407
    [157,   500] loss: 0.0021514
    [156 epoch] Accuracy of the network on the validation images: 86 %
    [158,   250] loss: 0.0023955
    [158,   500] loss: 0.0020414
    [157 epoch] Accuracy of the network on the validation images: 85 %
    [159,   250] loss: 0.0022236
    [159,   500] loss: 0.0028842
    [158 epoch] Accuracy of the network on the validation images: 86 %
    [160,   250] loss: 0.0023741
    [160,   500] loss: 0.0027341
    [159 epoch] Accuracy of the network on the validation images: 86 %
    [161,   250] loss: 0.0018917
    [161,   500] loss: 0.0024527
    [160 epoch] Accuracy of the network on the validation images: 86 %
    [162,   250] loss: 0.0029795
    [162,   500] loss: 0.0015014
    [161 epoch] Accuracy of the network on the validation images: 86 %
    [163,   250] loss: 0.0018990
    [163,   500] loss: 0.0027845
    [162 epoch] Accuracy of the network on the validation images: 86 %
    [164,   250] loss: 0.0027757
    [164,   500] loss: 0.0020061
    [163 epoch] Accuracy of the network on the validation images: 86 %
    [165,   250] loss: 0.0025747
    [165,   500] loss: 0.0032532
    [164 epoch] Accuracy of the network on the validation images: 86 %
    [166,   250] loss: 0.0030257
    [166,   500] loss: 0.0028665
    [165 epoch] Accuracy of the network on the validation images: 86 %
    [167,   250] loss: 0.0028003
    [167,   500] loss: 0.0025854
    [166 epoch] Accuracy of the network on the validation images: 86 %
    [168,   250] loss: 0.0037320
    [168,   500] loss: 0.0021234
    [167 epoch] Accuracy of the network on the validation images: 86 %
    [169,   250] loss: 0.0021324
    [169,   500] loss: 0.0028804
    [168 epoch] Accuracy of the network on the validation images: 86 %
    [170,   250] loss: 0.0020898
    [170,   500] loss: 0.0025412
    [169 epoch] Accuracy of the network on the validation images: 86 %
    [171,   250] loss: 0.0023761
    [171,   500] loss: 0.0029430
    [170 epoch] Accuracy of the network on the validation images: 86 %
    [172,   250] loss: 0.0020323
    [172,   500] loss: 0.0028294
    [171 epoch] Accuracy of the network on the validation images: 85 %
    [173,   250] loss: 0.0015253
    [173,   500] loss: 0.0020818
    [172 epoch] Accuracy of the network on the validation images: 86 %
    [174,   250] loss: 0.0026538
    [174,   500] loss: 0.0026940
    [173 epoch] Accuracy of the network on the validation images: 86 %
    [175,   250] loss: 0.0022194
    [175,   500] loss: 0.0022911
    [174 epoch] Accuracy of the network on the validation images: 86 %
    [176,   250] loss: 0.0022467
    [176,   500] loss: 0.0022120
    [175 epoch] Accuracy of the network on the validation images: 86 %
    [177,   250] loss: 0.0026406
    [177,   500] loss: 0.0028550
    [176 epoch] Accuracy of the network on the validation images: 86 %
    [178,   250] loss: 0.0022597
    [178,   500] loss: 0.0028486
    [177 epoch] Accuracy of the network on the validation images: 86 %
    [179,   250] loss: 0.0020362
    [179,   500] loss: 0.0025921
    [178 epoch] Accuracy of the network on the validation images: 86 %
    [180,   250] loss: 0.0021241
    [180,   500] loss: 0.0028593
    [179 epoch] Accuracy of the network on the validation images: 86 %
    [181,   250] loss: 0.0018245
    [181,   500] loss: 0.0032487
    [180 epoch] Accuracy of the network on the validation images: 86 %
    [182,   250] loss: 0.0025811
    [182,   500] loss: 0.0023134
    [181 epoch] Accuracy of the network on the validation images: 85 %
    [183,   250] loss: 0.0016276
    [183,   500] loss: 0.0021857
    [182 epoch] Accuracy of the network on the validation images: 85 %
    [184,   250] loss: 0.0037570
    [184,   500] loss: 0.0017745
    [183 epoch] Accuracy of the network on the validation images: 86 %
    [185,   250] loss: 0.0030020
    [185,   500] loss: 0.0020985
    [184 epoch] Accuracy of the network on the validation images: 86 %
    [186,   250] loss: 0.0025530
    [186,   500] loss: 0.0030716
    [185 epoch] Accuracy of the network on the validation images: 86 %
    [187,   250] loss: 0.0018812
    [187,   500] loss: 0.0026140
    [186 epoch] Accuracy of the network on the validation images: 86 %
    [188,   250] loss: 0.0035114
    [188,   500] loss: 0.0022442
    [187 epoch] Accuracy of the network on the validation images: 86 %
    [189,   250] loss: 0.0028399
    [189,   500] loss: 0.0030017
    [188 epoch] Accuracy of the network on the validation images: 86 %
    [190,   250] loss: 0.0021538
    [190,   500] loss: 0.0022238
    [189 epoch] Accuracy of the network on the validation images: 86 %
    [191,   250] loss: 0.0022629
    [191,   500] loss: 0.0024178
    [190 epoch] Accuracy of the network on the validation images: 86 %
    [192,   250] loss: 0.0026031
    [192,   500] loss: 0.0018577
    [191 epoch] Accuracy of the network on the validation images: 86 %
    [193,   250] loss: 0.0019058
    [193,   500] loss: 0.0031233
    [192 epoch] Accuracy of the network on the validation images: 86 %
    [194,   250] loss: 0.0022766
    [194,   500] loss: 0.0023194
    [193 epoch] Accuracy of the network on the validation images: 86 %
    [195,   250] loss: 0.0022375
    [195,   500] loss: 0.0021492
    [194 epoch] Accuracy of the network on the validation images: 86 %
    [196,   250] loss: 0.0034841
    [196,   500] loss: 0.0027367
    [195 epoch] Accuracy of the network on the validation images: 86 %
    [197,   250] loss: 0.0019502
    [197,   500] loss: 0.0020695
    [196 epoch] Accuracy of the network on the validation images: 86 %
    [198,   250] loss: 0.0023790
    [198,   500] loss: 0.0026829
    [197 epoch] Accuracy of the network on the validation images: 86 %
    [199,   250] loss: 0.0029993
    [199,   500] loss: 0.0019856
    [198 epoch] Accuracy of the network on the validation images: 86 %
    [200,   250] loss: 0.0030558
    [200,   500] loss: 0.0011285
    [199 epoch] Accuracy of the network on the validation images: 87 %
    Finished Training



```python
import time
start_time = time.asctime(time.localtime(time.time()))
print ("Train Finish at: ", start_time)
```

    Train Finish at:  Sat Dec 12 00:24:28 2020



```python
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```

    Accuracy of the network on the 10000 test images: 86 %



```python
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
                
        for i in range(labels.shape[0]):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
```

    Accuracy of plane : 89 %
    Accuracy of   car : 88 %
    Accuracy of  bird : 81 %
    Accuracy of   cat : 74 %
    Accuracy of  deer : 86 %
    Accuracy of   dog : 79 %
    Accuracy of  frog : 91 %
    Accuracy of horse : 91 %
    Accuracy of  ship : 92 %
    Accuracy of truck : 88 %


