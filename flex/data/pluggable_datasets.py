from enum import Enum, EnumMeta


class PluggableDataset(EnumMeta):
    def __contains__(self, item):
        try:
            self(item)
        except ValueError:
            return False
        else:
            return True


class PluggableTorchtext(Enum, metaclass=PluggableDataset):
    from torchtext import datasets

    """Class containing all the pluggable datasets to a FlexDataObject without any preprocessing needed.

    Any other dataset from the TorchText library will need further preprocessing.

    Args:
        Enum (enum): torchtext class for each dataset than can be accepted on our platform.
    """

    AG_NEWS_PT = datasets.AG_NEWS.__name__
    AMAZON_REVIEW_FULL_PT = datasets.AmazonReviewFull.__name__
    AMAZON_REVIEW_POLARITY_PT = datasets.AmazonReviewPolarity.__name__
    DBPEDIA_PT = datasets.DBpedia.__name__
    YAHOOANSWERS_PT = datasets.YahooAnswers.__name__
    YELPREVIEWFULL_PT = datasets.YelpReviewFull.__name__
    YELPREVIEWPOLARITY_PT = datasets.YelpReviewPolarity.__name__


class PluggableTorchvision(Enum, metaclass=PluggableDataset):
    from torchvision import datasets

    """Class containing all the pluggable datasets to a FlexDataObject without any preprocessing needed.

    Any other dataset from the Torchvision library will need further preprocessing.

    Args:
        Enum (enum): torchvision class for each dataset than can be accepted on our platform.
    """
    WIDERFace_PT = datasets.WIDERFace.__name__
    Food101_PT = datasets.Food101.__name__
    CelebA_PT = datasets.CelebA.__name__
    CLEVRClassification_PT = datasets.CLEVRClassification.__name__
    Country211_PT = datasets.Country211.__name__
    FGVCAircraft_PT = datasets.FGVCAircraft.__name__
    GTSRB_PT = datasets.GTSRB.__name__
    Kitti_PT = datasets.Kitti.__name__
    Flowers102_PT = datasets.Flowers102.__name__
    StanfordCars_PT = datasets.StanfordCars.__name__
    LFWPeople_PT = datasets.LFWPeople.__name__
    Caltech256_PT = datasets.Caltech256.__name__
    EuroSAT_PT = datasets.EuroSAT.__name__
    CIFAR10_PT = datasets.CIFAR10.__name__
    CIFAR100_PT = datasets.CIFAR100.__name__
    MNIST_PT = datasets.MNIST.__name__
    SUN397_PT = datasets.SUN397.__name__
    SEMEION_PT = datasets.SEMEION.__name__
    Omniglot_PT = datasets.Omniglot.__name__
    KMNIST_PT = datasets.KMNIST.__name__
    FashionMNIST_PT = datasets.FashionMNIST.__name__
    OxfordIIITPet_PT = datasets.OxfordIIITPet.__name__
    STL10_PT = datasets.STL10.__name__
    PCAM_PT = datasets.PCAM.__name__
    Caltech101_PT = datasets.Caltech101.__name__
    QMNIST_PT = datasets.QMNIST.__name__
    SVHN_PT = datasets.SVHN.__name__
    DTD_PT = datasets.DTD.__name__
    USPS_PT = datasets.USPS.__name__
    RenderedSST2_PT = datasets.RenderedSST2.__name__
    INaturalist_PT = datasets.INaturalist.__name__
    EMNIST_PT = datasets.EMNIST.__name__


# class PluggableHuggingFace(Enum, metaclass=PluggableDataset):
#     """Class containing some datasets that can be loaded to FLEXible. Other datasets
#     can be plugged in, but it requires a special configuration, i.e., glue-cola. This
#     is more about the user using correctly the arguments on the load_dataset function
#     from huggingface datasets than a problem of our platform, so the user can easy-use
#     other datasets.

#     Args:
#         Enum (enum): Tuple containing name, X_columns and y_columns to use in the
#         load_dataset function.
#     """

#     IMDB_HF = ("imdb", "text", "label")
#     SQUAD_HF = ("squad", ["context", "question"], "answers")
#     APPREVIEWS_HF = ("app_reviews", "review", "star")
#     AMAZON_POLARITY_HF = ("amazon_polarity", ["title", "content"], "label")


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
