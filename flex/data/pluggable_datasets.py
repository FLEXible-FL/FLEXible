"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI).

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
from enum import Enum, EnumMeta


class PluggableDataset(EnumMeta):
    def __contains__(self, item):
        return item in list(self.members())


class PluggableDatasetString(EnumMeta):
    def __contains__(self, item):
        return item in [k.value[0] for k in self.__members__.values()]


class PluggableTorchtext(Enum, metaclass=PluggableDataset):
    """Class containing all the pluggable datasets to a Dataset without any preprocessing needed.

    Any other dataset from the TorchText library will need further preprocessing.

    Args:
    -----
        Enum (enum): torchtext class for each dataset than can be accepted on our platform.
    """

    def members():
        from torchtext import datasets

        yield datasets.AG_NEWS.__name__
        yield datasets.AmazonReviewFull.__name__
        yield datasets.AmazonReviewPolarity.__name__
        yield datasets.DBpedia.__name__
        yield datasets.YahooAnswers.__name__
        yield datasets.YelpReviewFull.__name__
        yield datasets.YelpReviewPolarity.__name__


class PluggableTorchvision(Enum, metaclass=PluggableDataset):

    """Class containing all the pluggable datasets to a Dataset without any preprocessing needed.

    Any other dataset from the Torchvision library will need further preprocessing.

    Args:
    -----
        Enum (enum): torchvision class for each dataset than can be accepted on our platform.
    """

    def members():
        from torchvision import datasets

        yield datasets.WIDERFace.__name__
        yield datasets.Food101.__name__
        yield datasets.CelebA.__name__
        yield datasets.CLEVRClassification.__name__
        yield datasets.Country211.__name__
        yield datasets.FGVCAircraft.__name__
        yield datasets.GTSRB.__name__
        yield datasets.Kitti.__name__
        yield datasets.Flowers102.__name__
        yield datasets.StanfordCars.__name__
        yield datasets.LFWPeople.__name__
        yield datasets.Caltech256.__name__
        yield datasets.EuroSAT.__name__
        yield datasets.CIFAR10.__name__
        yield datasets.CIFAR100.__name__
        yield datasets.MNIST.__name__
        yield datasets.SUN397.__name__
        yield datasets.SEMEION.__name__
        yield datasets.Omniglot.__name__
        yield datasets.KMNIST.__name__
        yield datasets.FashionMNIST.__name__
        yield datasets.OxfordIIITPet.__name__
        yield datasets.STL10.__name__
        yield datasets.PCAM.__name__
        yield datasets.Caltech101.__name__
        yield datasets.QMNIST.__name__
        yield datasets.SVHN.__name__
        yield datasets.DTD.__name__
        yield datasets.USPS.__name__
        yield datasets.RenderedSST2.__name__
        yield datasets.INaturalist.__name__
        yield datasets.EMNIST.__name__


class PluggableHuggingFace(Enum, metaclass=PluggableDatasetString):
    """Class containing some datasets that can be loaded to FLEXible. Other datasets
    can be plugged in, but it requires a special configuration, i.e., glue-cola. This
    is more about the user using correctly the arguments on the load_dataset function
    from huggingface datasets than a problem of our platform, so the user can easy-use
    other datasets.

    We show some example datasets that can be loaded using the function
    FedDataDistribution.from_config_with_huggingface_dataset just giving a config
    and the string associated to each dataset from the Enum defined.

    We selected this dataset as we can automatice the process of loading this datasets,
    but our framework support almost all the datasets, as they can be loaded as numpy
    arrays. We only show supports to this datasets as we can load the dataset
    as follows: dataset = load_dataset(name, split='train').

    There are some datasets that need extra parameters like the version of the dataset,
    or that don't have any split. This must be used by the user previously to load
    the dataset into FLEXible, but it will be easy and fast, as the user just
    need to select the X_train-y_train as np.arrays.

    Args:
    -----
        Enum (enum): Tuple containing name, X_columns and y_columns to use in the
        load_dataset function.
    """

    IMDB_HF = ("imdb", "text", "label")
    AG_NEWS_HF = ("ag_news", "text", "label")
    TWEET_EVAL_EMOJI_HF = ("tweet_eval", "text", "label")  # We support all subsets
    ROTTEN_TOMATOES_HF = ("rotten_tomatoes", "text", "label")
    GLUE_COLA_HF = ("glue", "sentence", "label")
    FINANCIAL_PHRASEBANK_HF = (
        "financial_phrasebank",
        "sentence",
        "label",
    )  # We support all subsets"
    SQUAD_HF = ("squad", ["context", "question"], "answers")
    APPREVIEWS_HF = ("app_reviews", "review", "star")
    AMAZON_POLARITY_HF = ("amazon_polarity", ["title", "content"], "label")


# class PluggableDatasetsTensorFlowText(Enum, metaclass=PluggableDataset):
#     """Class containing some datasets that can be loaded to FLEXible. Other datasets
#     can be plugged in, but it requires a special configuration, i.e., glue-cola. This
#     is more about the user using correctly the arguments on the load_dataset function
#     from huggingface datasets than a problem of our platform, so the user can easy-use
#     other datasets.

#     Args:
#         Enum (enum): Tuple containing name, X_columns and y_columns to use in the
#         load_dataset function.
#     """

#     AG_NEWS_TF = ("ag_news_subset", ["title", "description"], ["label"])
#     GLUE_TF = ("glue", ["sentence"], ["label"])
#     ASSET_TF = ("asset", ["original"], ["simplifications"])
#     SQUAD_TF = ("squad", ["title", "question", "context"], ["answers"])  # Wonk work
#     COQA_TF = ("coqa", ["questions", "source", "story"], ["answers"])  # Won't work
