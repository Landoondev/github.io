# TensorBoard & Summary

```python
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

writer = SummaryWriter()

model = torchvision.models.vgg11()
dummy_input = torch.rand(2, 3, 256, 256)  # 假设输入2张3*256*256的图片
writer.add_graph(model, dummy_input)
```



```python
tensorboard --logdir=runs
```



```python
from torchsummary import summary

summary(model, (3, 64, 64))
```

