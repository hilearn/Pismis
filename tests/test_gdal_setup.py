import unittest
from osgeo import gdal


class TestGdalVersion(unittest.TestCase):
    def test_gdal_version(self):
        self.assertEquals(gdal.__version__, '2.1.3')
