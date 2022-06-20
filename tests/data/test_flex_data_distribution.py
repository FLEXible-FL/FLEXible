import unittest

import pytest

from flex.data.flex_data_distribution import FlexDataDistribution


class TestFlexDataDistribution(unittest.TestCase):
    def test_init_method_does_not_work(self):
        with pytest.raises(AssertionError):
            FlexDataDistribution()
