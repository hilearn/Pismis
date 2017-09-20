import unittest
from osgeo import gdal


class TestGdalVersion(unittest.TestCase):
    def test_gdal_version(self):
        self.assertEquals(gdal.__version__, '2.1.3')

    def test_gdal_work(self):
        image = gdal.Open("tests/data/T38TML_20170830T074611_B01.jp2")
        self.assertIsNotNone(image)
