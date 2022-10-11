"""
TorchText Datasets: 'AG_NEWS', 'AmazonReviewFull', 'AmazonReviewPolarity', 'DBpedia', 'YahooAnswers', 'YelpReviewFull', 'YelpReviewPolarity'.
"""

from enum import Enum

from torchtext import datasets


class PluggableDatasetsTorchText(Enum):
    """Class containing all the pluggable datasets to a FlexDataObject without any preprocessing needed.

    Any other dataset from the TorchText library will need further preprocessing.

    Args:
        Enum (enum): torchtext class for each dataset than can be accepted on our platform.
    """

    AG_NEWS_PT = datasets.AG_NEWS
    AMAZON_REVIEW_FULL_PT = datasets.AmazonReviewFull
    AMAZON_REVIEW_POLARITY_PT = datasets.AmazonReviewPolarity
    DBPEDIA_PT = datasets.DBpedia
    YAHOOANSWERS_PT = datasets.YahooAnswers
    YELPREVIEWFULL_PT = datasets.YelpReviewFull
    YELPREVIEWPOLARITY_PT = datasets.YelpReviewPolarity


class PluggableDatasetsHuggingFace(Enum):
    """Class containing some datasets that can be loaded to FLEXible. Other datasets
    can be plugged in, but it requires a special configuration, i.e., glue-cola. This
    is more about the user using correctly the arguments on the load_dataset function
    from huggingface datasets than a problem of our platform, so the user can easy-use
    other datasets.

    Args:
        Enum (enum): Tuple containing name, X_columns and y_columns to use in the
        load_dataset function.
    """

    IMDB_HF = ("imdb", "text", "label")
    SQUAD_HF = ("squad", ["context", "question"], "answers")
    APPREVIEWS_HF = ("app_reviews", "review", "star")
    AMAZON_POLARITY_HF = ("amazon_polarity", ["title", "content"], "label")
