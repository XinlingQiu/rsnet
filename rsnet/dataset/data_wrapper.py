import rasterio
import numpy as np

from ..utils import pair


class RasterDataIterator(object):
    """Dataset wrapper for remote sensing data.

    Args:
        fname:
        patch_size:
        slide_step:
        pad_size:
        band_index:
    """
    def __init__(self,
                 fname,
                 patch_size=512,
                 slide_step=512,
                 pad_size=0,
                 band_index=(1, 2, 3)):
        self.fname = fname
        # with rasterio.open(fname) as src:
        #     self.profile = src.profile

        self._band = rasterio.open(fname)

        self.patch_size = pair(patch_size)
        self.slide_step = pair(slide_step)
        self.pad_size = pair(pad_size)

        total_band_index = [i + 1 for i in range(self.count)]
        assert set(band_index).issubset(set(total_band_index))
        self.band_index = band_index

        self.tile_ids = self.get_tile_info()

        self.start = 0
        self.end = len(self)

    def __del__(self):
        self._band.close()

    def get_tile_info(self):
        left, top = 0, 0
        width, height = self.width, self.height
        left_top_xy = []  # left-top corner coordinates (xmin, ymin)
        while left < width:
            if left + self.patch_size[0] >= width:
                left = max(width - self.patch_size[0], 0)
            top = 0
            while top < height:
                if top + self.patch_size[1] >= height:
                    top = max(height - self.patch_size[1], 0)
                # right = min(left + self.patch_size[0], width - 1)
                # bottom = min(top + self.patch_size[1], height - 1)
                # save
                left_top_xy.append((left, top))
                if top + self.patch_size[1] >= height:
                    break
                else:
                    top += self.slide_step[1]

            if left + self.patch_size[0] >= width:
                break
            else:
                left += self.slide_step[0]

        return left_top_xy

    def sample(self, x, y):
        """Get the values of dataset at certain positions.
        """
        xmin, ymin = x, y
        xsize, ysize = self.patch_size
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
        tile_window = rasterio.windows.Window(xmin, ymin, xsize, ysize)

        # with rasterio.open(self.image_file) as src:
        #     bands = [src.read(k, window=tile_window) for k in self.band_index]
        #     tile_image = np.stack(bands, axis=-1)
        bands = [
            self._band.read(k, window=tile_window) for k in self.band_index
        ]
        tile_image = np.stack(bands, axis=-1)

        img = np.zeros(
            (self.patch_size[0] + 2 * xpad, self.patch_size[0] + 2 * ypad,
             len(self.band_index)),
            dtype=tile_image.dtype)

        img[top:top + ysize, left:left + xsize] = tile_image

        return img

    def __getitem__(self, idx):
        x, y = self.tile_ids[idx]
        img = self.sample(x, y)

        return img, (x, y)

    def __iter__(self):
        return self

    def __next__(self):
        if self.start < self.end:
            item = self.__getitem__(self.start)
            self.start += 1
            return item
        else:
            raise StopIteration

    def __len__(self):
        return len(self.tile_ids)

    @property
    def width(self):
        return self._band.width

    @property
    def height(self):
        return self._band.height

    @property
    def count(self):
        """Band counts."""
        return self._band.count
