"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flex.data.dataset import Dataset
from flex.data.fed_dataset import FedDataset
from flex.data.fed_dataset_config import FedDatasetConfig
from flex.data.fed_data_distribution import FedDataDistribution
from flex.data.preprocessing_utils import normalize
from flex.data.preprocessing_utils import one_hot_encoding
from flex.data.pluggable_datasets import PluggableTorchtext
from flex.data.pluggable_datasets import PluggableTorchvision
from flex.data.pluggable_datasets import PluggableHuggingFace
from flex.data.lazy_indexable import LazyIndexable