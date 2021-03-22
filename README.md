# RSNet

Python library to work with geospatial raster and vector data for
deep learning.

RSNet is designed to make it easier for the deep learning
researchers to handle the remote sensing data.



## QuickStart

1. perform model prediction for large remote sensing image.

```python
from rsnet.dataset import RasterSampleDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T

tsf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
])
ds = RasterSampleDataset('example.tif',
                         win_size=512,
                         step_size=512,
                         pad_size=128,
                         band_index=(3, 2, 1),
                         transform=tsf)
# Deep learning model predict
loader = DataLoader(ds,
                    batch_size=1,
                    num_workers=0,
                    shuffle=False,
                    drop_last=False)
for img, xoff, yoff in loader:
    with torch.no_grad():
        result = model(img)
```

2. split large image into tile image.

```python
from rsnet.converter import RasterDataSpliter

ds_spliter = RasterDataSpliter('example.tif',
                            win_size=512,
                            step_size=512)
ds_spliter.run('/path/to/output', progress=True)
```

