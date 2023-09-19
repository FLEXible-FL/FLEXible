# Copyright 2023 Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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