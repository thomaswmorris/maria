import numpy as np
from dask import array as da


class Field:
    def __init__(self, data: float, dtype: type = np.float32):
        self.dtype = dtype
        self.data = data

    @property
    def data(self):
        return self._data + self._offset[..., None]

    @data.setter
    def data(self, value):
        self._offset = value.mean(axis=-1)  # average over time axis
        self._data = da.asarray(value - self._offset[..., None]).astype(self.dtype)

    def __getitem__(self, key):
        return Field(data=self.data.__getitem__(key), dtype=self.dtype)

    def __repr__(self):
        return f"Field({self.data.__repr__()})"
