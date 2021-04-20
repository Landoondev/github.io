# ResNeSt50 PyTorch

（1）ResNeSt 基于 ResNet，如下是 PyTorch 官方实现的 ResNet50 接口。

```python
def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
```

ResNeSt 的接口如下：

```python
def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
    	pass
    return model
```

ResNeSt 的本质是基于 ResNet 加入 Split-Attention block（ResNeSt block）。

Bottleneck 毫无疑问是需要修改的，加入 ResNeSt block。

[3, 4, 6, 3] 与 ResNet 一样。其中的 50 层是这样数的。

- 每个 Bottleneck 中有 3 个卷积层，因此 (3+4+6+3) * 3 = 48。
- 在加上第一个层的 7×7，以及最后一个 FC 层，总共 50 层。

不过看实际代码，ResNeSt 在进入 Bottleneck 前会多出来几层。



多出了一些参数：

- (a) `radix`：Split ？
- (b) `group`：分成 group 组进行卷积（分组卷积）
- (c) `bottleneck_width=64`：？
- (d) `deep_stem`：？
- (e) `stem_width`：？
- (f) `avg_down`：？
- (g) `avd/avd_first`：？



（2）深入 ResNet 内部去看。

```python
class ResNet(nn.Module):
	"""ResNet Variants"""
    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
             num_classes=1000, dilated=False, dilation=1,
             deep_stem=False, stem_width=64, avg_down=False,
             rectified_conv=False, rectify_avg=False,
             avd=False, avd_first=False,
             final_drop=0.0, dropblock_prob=0,
             last_gamma=False, norm_layer=nn.BatchNorm2d):
```

只关心的参数：

- `block=Bottleneck`
- `layers=[3, 4, 6, 3]`
- `radix=2`
- `group=1`
- `bottleneck=64`
- `deep_stem=True`：使用 3 个 3×3 的替代一个 7×7 的。 
- `stem_width=64`：
- `avg_down=True`：在 _make_layer 函数中，控制特征图的大小要不要减半。
- 

```python
class Bottleneck(nn.Module):
	"""ResNet Bottleneck"""
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False):
```



## SplAtConv2d

输入的图片 size 为 (3×64×64)，第一个 Bottleneck 中，输入 SplAtConv2d 的特征图大小为 **64×16×16**。

（1）Conv

```python
# @in_channels=64
# @channels*radix=128
# @kernel_size=3
# @stride=1
# @padding=1
# @dilation=1
# @groups=2
# @bias=True
self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
```

```python
# in_channels=64, channels*radix=128, groups=2
conv = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1,
               groups=2, bias=True)
input = torch.randn(2, 64, 16, 16)
out = conv(input)
out.shape # Out: torch.Size([2, 128, 16, 16])
```

默认 Split=2（radix 参数=2）。

64×16×16 的特征图进入两条分支，每个分支中进行分组卷积（分组数为 2，相当于 cardinality=2）。



（2）

