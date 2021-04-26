import fiona
import rasterio
from rasterio import features


def rasterize(file, output, driver, like, nodata=0, **kwargs):
    """Rasterize vector into a new raster.
    """
    with fiona.open(file) as src:
        schema = src.schema
        shapes = ((f['geometry'], int(f['properties']['id'])) for f in src)

        with rasterio.open(like) as ref:
            out_shape = ref.shape
            transform = ref.transform

        meta = {
            'count': 1,
            'crs': ref.crs,
            'width': ref.width,
            'height': ref.height,
            'transform': transform,
            'driver': driver,
            'dtype': rasterio.uint8,
            'nodata': nodata,
            'compress': 'lzw',
            'interleave': 'pixel',
            **kwargs
        }

        image = features.rasterize(shapes,
                                   out_shape=out_shape,
                                   fill=0,
                                   transform=transform)
        with rasterio.open(output, 'w', **meta) as dst:
            dst.write(image, indexes=1)
