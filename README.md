# LandmarkConv
Landmark Feature Convolutional Networks

### installation
#### requirements
- Linux with Python 3.6
- CUDA support 
- gcc 4.9 or above

```bash
# git clone git://github.com/hbb1/landmarkconv
cd lanmarkconv/lib/layers
make
```

### Usage
For convenience, LandmarkConvs are implemented as subclass of torch.nn.Conv2d, so just use it like standard convolution in Pytorch.

```Python
from .lib.layers.conv4 import PConv2d4
from .lib.layers.conv8 import PConv2d8
```