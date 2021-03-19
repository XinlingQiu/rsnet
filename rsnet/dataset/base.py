import os

import rasterio


class BaseRasterData:
    def __init__(self, fname):
        self.fname = fname
        self._band = rasterio.open(self.fname)

    def __del__(self):
        self._band.close()

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

    @property
    def crs(self):
        return self._band.crs

    @property
    def transform(self):
        """Transform matrix as the `affine.Affine`
        
        This transform maps pixel row/column coordinates to coordinates in the dataset’s coordinate reference system.
        
        affine.identity is returned if if the file does not contain transform
        """
        return self._band.transform

    @property
    def nodata(self):
        """
        Band nodata value, type depends on the image dtype; None if the nodata value is not specified
        """
        return self._band.nodata

    @property
    def res(self):
        """
        Spatial resolution (x_res, y_res) of the Band in X and Y directions of the georeferenced coordinate system,
        derived from tranaform. Normally is equal to (transform.a, - transform.e)
        """
        return self._band.res

    @property
    def shape(self):
        """
        The raster dimension as a Tuple (height, width)
        """
        return self.height, self.width

    @property
    def name(self):
        """
        Name of the file, without extension and the directory path
        """
        return os.path.basename(self._band.name).split('.')[0]

    @property
    def suffix(self):
        """
        Name of the file, without extension and the directory path
        """
        return os.path.basename(self._band.name).split('.')[1]

    @property
    def bounds(self):
        """
        Georeferenced bounds - bounding box in the CRS of the image, based on transform and shape
        
        Returns:
            `BoundingBox object
            <https://rasterio.readthedocs.io/en/latest/api/rasterio.coords.html#rasterio.coords.BoundingBox>`_:
            (left, bottom, right, top)
        """
        return self._band.bounds

    @property
    def meta(self):
        """
        The basic metadata of the associated rasterio DatasetReader
        """
        return self._band.meta

    @property
    def dtype(self):
        """
        Numerical type of the data stored in raster, according to numpy.dtype
        """
        return self._band.dtypes[0]
