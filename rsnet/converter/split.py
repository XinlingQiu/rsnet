import os.path as osp

import rasterio as rio
from affine import Affine

from ..dataset import RasterDataIterator
from ..utils import mkdir


def window_transform(window, transform):
    """Construct an affine transform matrix relative to a window.
    
    Args:
        window (Window): The input window.
        transform (Affine): an affine transform matrix.
    Returns:
        (Affine): The affine transform matrix for the given window
    """
    x, y = transform * (window.col_off, window.row_off)
    return Affine.translation(x - transform.c, y - transform.f) * transform


def split(fname, outpath, win_size, step_size, suffix_tmpl='_{}_{}'):
    ds = RasterDataIterator(fname,
                            win_size=win_size,
                            step_size=step_size,
                            pad_size=0)

    mkdir(outpath)
    basename = ds.name
    suffix = f'{suffix_tmpl}.{ds.suffix}'
    driver = ds.meta['driver']
    for i, (tile, win) in enumerate(ds, start=1):
        xoff, yoff = win.col_off, win.row_off
        width, height = win.width, win.height
        outfile = osp.join(outpath, basename + suffix.format(xoff, yoff))
        transform = window_transform(win, ds.transform)
        with rio.open(outfile,
                      'w',
                      driver=driver,
                      width=width,
                      height=width,
                      crs=ds.crs,
                      count=len(ds.band_index),
                      dtype=ds.dtype,
                      transform=transform) as dst:
            dst.write(tile.transpose(2, 0, 1))
