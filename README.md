# RSNet

Python library to work with geospatial raster and vector data for
deep learning.

RSNet is designed to make it easier for the deep learning
researchers to handle the remote sensing data.



## QuickStart

1. perform model prediction for large remote sensing image.

```python
from rsnet.dataset import RasterDataIterator

ds_iter = RasterDataIterator('example.tif',
                             win_size=512,
                             step_size=512,
                             pad_size=128,
                             band_index=(3, 2, 1))

# Deep learning model predict
for img, win in ds_iter:
    # convert img to tensor
    ...
    #
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

