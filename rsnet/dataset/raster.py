import os

import rasterio
import numpy as np

from ..utils import pair
from .base import BaseRasterData


class RasterSampleDataset(BaseRasterData):
    """Dataset wrapper for remote sensing data.

    Args:
        fname:
        win_size:
        step_size:
        pad_size:
        band_index:
    """
    def __init__(self,
                 fname,
                 win_size=512,
                 step_size=512,
                 pad_size=0,
                 band_index=None,
                 transform=None):
        super().__init__(fname=fname)
        self.win_size = pair(win_size)
        self.step_size = pair(step_size)
        self.pad_size = pair(pad_size)

        total_band_index = [i + 1 for i in range(self.count)]
        if band_index is None:
            self.band_index = total_band_index
        else:
            assert set(band_index).issubset(set(total_band_index))
            self.band_index = band_index

        self.window_ids = self.get_windows_info()
        self.transform = transform

        self.start = 0
        self.end = len(self)

    def get_windows_info(self):
        left, top = 0, 0
        width, height = self.width, self.height
        left_top_xy = []  # left-top corner coordinates (xmin, ymin)
        while left < width:
            if left + self.win_size[0] >= width:
                left = max(width - self.win_size[0], 0)
            top = 0
            while top < height:
                if top + self.win_size[1] >= height:
                    top = max(height - self.win_size[1], 0)
                # right = min(left + self.win_size[0], width - 1)
                # bottom = min(top + self.win_size[1], height - 1)
                # save
                left_top_xy.append((left, top))
                if top + self.win_size[1] >= height:
                    break
                else:
                    top += self.step_size[1]

            if left + self.win_size[0] >= width:
                break
            else:
                left += self.step_size[0]

        return left_top_xy

    def sample(self, x, y):
        """Get the values of dataset at certain positions.
        """
        xmin, ymin = x, y
        xsize, ysize = self.win_size
        xpad, ypad = self.pad_size

        xmin -= xpad
        ymin -= ypad
        left, top = 0, 0
        if xmin < 0:
            xmin = 0
            xsize += xpad
            left = xpad
        elif xmin + xsize + 2 * xpad > self.width:
            xsize += xpad
        else:
            xsize += 2 * xpad

        if ymin < 0:
            ymin = 0
            ysize += ypad
            top = ypad
        elif ymin + ysize + 2 * ypad > self.height:
            ysize += ypad
        else:
            ysize += 2 * ypad

        # col_off, row_off, width, height
        window = rasterio.windows.Window(xmin, ymin, xsize, ysize)

        # with rasterio.open(self.image_file) as src:
        #     bands = [src.read(k, window=tile_window) for k in self.band_index]
        #     tile_image = np.stack(bands, axis=-1)
        bands = [self._band.read(k, window=window) for k in self.band_index]
        tile_image = np.stack(bands, axis=-1)

        img = np.zeros(
            (self.win_size[0] + 2 * xpad, self.win_size[0] + 2 * ypad,
             len(self.band_index)),
            dtype=tile_image.dtype)

        img[top:top + ysize, left:left + xsize] = tile_image

        return img

    def __getitem__(self, idx):
        x, y = self.window_ids[idx]
        img = self.sample(x, y)
        if self.transform is not None:
            img = self.transform(img)

        return img, x, y

    def __len__(self):
        return len(self.window_ids)

    @property
    def step(self):
        return self.step_size

    @property
    def pad(self):
        return self.pad_size
