import os
import zipfile
from hashlib import md5

from sultan.api import Sultan
from tqdm import tqdm

EMNIST_URL = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip"
EMNIST_MD5 = "1bbb49fdf3462bb70c240eac93fff0e4"
EMNIST_FILE = "mnist.zip"
EMNIST_DIGITS = "emnist-digits.mat"
EMNIST_LETTERS = "emnist-letters.mat"
EMNIST_MNIST = "emnist-mnist.mat"


def check_hash(filename: str, md5_hash: str) -> bool:
    """Function that computes the md5 hash of a file and compares it
        with a given one, ensuring that the file corresponds to the given md5 hash

    Args:
        filename (str): path to file which will be used to compute a md5 hash
        md5_hash (str): md5 hash to compare with the one compute using filename

    Returns:
        bool: whether the given file has the same hash as the one provided
    """
    with open(filename, "rb") as file_to_check:
        data = file_to_check.read()
        md5_returned = md5(data).hexdigest()
    return md5_returned == md5_hash


def check_file_exists(filename: str) -> bool:
    """Function that checks if a given file exits or not.

    Args:
        filename (str): Filename to check.

    Returns:
        bool: True/False if the file exits or not.
    """
    return os.path.isfile(filename)


def check_dir_exists(filename: str) -> bool:
    """Function that checks if a given directory exits or not.

    Args:
        filename (str): Directory to check.

    Returns:
        bool: True/False if the directory exits or not.
    """
    return os.path.exists(filename)


def extract_zip(filename: str, output: bool = True):
    """Function that extract a zip file.

    Args:
        filename (str): Directory to check.
        output (bool): Whether to output the paths of the extracted files

    Returns:
        bool: True/False if the directory exits or not.
    """
    with zipfile.ZipFile(filename, "r") as ref:
        ref.extractall()
        if output:
            return ref.namelist()


def download_file(url: str, filename: str, out_dir: str = "."):
    """Function that downloads a file from a url and stores it in out_dir
        with name filename

    Args:
        url (str): url to download the file
        filename (str): name used to store the downloaded file
        out_dir (str, optional): directory where the downloaded file will be stored. Defaults to ".".

    """
    additional_args = ()
    out_path = os.path.join(out_dir, filename)
    additional_args = ("-#", "-L", "--output", out_path)
    with Sultan.load() as s:
        result = s.curl(url, *additional_args).run(streaming=True)

        def generator():
            while True:
                complete = result.is_complete
                if complete:
                    break
                yield from result.stderr

        pbar = tqdm(generator())
        for i in pbar:
            pbar.set_description(i)


# Example: download_dataset(MNIST_URL, MNIST_FILE, MNIST_MD5, extract=True)
def download_dataset(
    url: str,
    filename: str,
    md5_hash: str,
    out_dir: str = ".",
    extract: bool = False,
    max_trials: int = 3,
    output: bool = True,
):
    """Function that download a dataset given an URL.

    Args:
        url (str): url to download the file
        filename (str): name used to store the downloaded file
        md5_hash (str): hash used to ensure the integrity of the downloaded file
        out_dir (str, optional): directory where the downloaded file will be stored. Defaults to ".".
        extract (bool, optional): Select to whether to extract the data or not. Defaults to False.
        max_trials (int, optional): Max number of trials to download the dataset. Defaults to 3.
        output (bool, optional): whether to return a list with the paths of the downloaded/extractred files. Defaults to True.

    Raises:
        ValueError: Raise an error if it fails downloading the dataset or the given md5 hash is not correct.
    """
    full_path = os.path.join(out_dir, filename)
    check_dir_exists(out_dir)
    i = 0
    while not (check_file_exists(full_path) and check_hash(full_path, md5_hash)):
        download_file(url, filename, out_dir)
        i += 1
        if i > max_trials:
            raise ValueError(
                "Either we are unable the download the file or the provided md5 hash is not correct."
            )
    if extract:
        extracted_files = extract_zip(full_path, output=output)
    if output:
        return (
            [os.path.join(out_dir, i) for i in extracted_files]
            if extract
            else full_path
        )
