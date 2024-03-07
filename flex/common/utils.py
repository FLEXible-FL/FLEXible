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
import inspect
import os
import zipfile
from typing import Callable

import gdown
from sultan.api import Sultan
from tqdm import tqdm

EMNIST_DIGITS_URL = "https://drive.google.com/file/d/1fl9fRPPxTUxnC56ACzZ8JiLiew0SMFwt/view?usp=share_link"
EMNIST_DIGITS_MD5 = "5a18b33e88e3884e79f8b2d6274564d7"
EMNIST_DIGITS_FILE = "emnist-digits.mat"

EMNIST_LETTERS_URL = "https://drive.google.com/file/d/1KpwKUfx5L8zN0gPyrEuCEDFnC0OlEOoM/view?usp=share_link"
EMNIST_LETTERS_MD5 = "b9eddc3e325dee05b65fb21ee45da52f"
EMNIST_LETTERS_FILE = "emnist-letters.mat"

REDDIT_URL = (
    "https://drive.google.com/file/d/1S5R11eDR6g9Lkb02Ht6zmP2W-ZYqtS5U/view?usp=sharing"
)
REDDIT_MD5 = "308519a1945cf9268707d0aeba2b1c96"
REDDIT_FILE = "reddit_leaf.zip"

SHAKESPEARE_URL = "https://drive.google.com/file/d/1YkFqXqvN1s1JE1lEQ5YHb-aBPgmQ2IIk/view?usp=share_link"
SHAKESPEARE_MD5 = "a75e79f9ecebe5118314227a9cd19bc9"
SHAKESPEARE_FILE = "shakespeare_leaf.zip"


def check_integrity(filename: str, md5_hash: str) -> bool:
    """Function that computes the md5 hash of a file and compares it
        with a given one, ensuring that the file corresponds to the given md5 hash.

    Args:
    -----
        filename (str): path to file which will be used to compute a md5 hash
        md5_hash (str): md5 hash to compare with the one compute using filename

    Returns:
    --------
        bool: whether the given file has the same hash as the one provided
    """
    with Sultan.load() as s:
        try:
            result = s.md5("-q", filename).run()
        except Exception:
            result = s.md5sum(filename).pipe().cut("-f", "1", "-d", '" "').run()
    computed_md5 = result.stdout[0]
    return computed_md5 == md5_hash


def check_file_exists(filename: str) -> bool:
    """Function that checks if a given file exits or not.

    Args:
    -----
        filename (str): Filename to check.

    Returns:
    --------
        bool: True/False if the file exits or not.
    """
    return os.path.isfile(filename)


def check_dir_exists(filename: str) -> bool:
    """Function that checks if a given directory exits or not.

    Args:
    -----
        filename (str): Directory to check.

    Returns:
    --------
        bool: True/False if the directory exits or not.
    """
    return os.path.exists(filename)


def extract_zip(filename: str, output: bool = True):
    """Function that extract a zip file. If files are already extracted, it skips
    extracting them.

    Args:
    -----
        filename (str): Directory to check.
        output (bool): Whether to output the paths of the extracted files

    Returns:
    --------
        bool: True/False if the directory exits or not.
    """
    base_dir = "/".join(filename.split("/")[:-1])
    with zipfile.ZipFile(filename, "r", allowZip64=True) as zip_file:
        for member in zip_file.namelist():
            extracted_file_path = f"{base_dir}/{member}"
            if not (
                os.path.isfile(extracted_file_path)
                or os.path.exists(extracted_file_path)
            ):
                zip_file.extract(member, base_dir)
        if output:
            return [f"{base_dir}/{member}" for member in zip_file.namelist()]


def download_file(url: str, filename: str, out_dir: str = "."):
    """Function that downloads a file from a url and stores it in out_dir
        with name filename.

    Args:
    -----
        url (str): url to download the file
        filename (str): name used to store the downloaded file
        out_dir (str, optional): directory where the downloaded file will be stored. Defaults to ".".

    """
    additional_args = ()
    out_path = os.path.join(out_dir, filename)
    if "drive.google" not in url:
        additional_args = ("-#", "-L", "--output", out_path)
        with Sultan.load() as s:
            try:
                result = s.curl(url, *additional_args).run(streaming=True)
            except Exception:
                result = s.wget(url, "-O", out_path).run(streaming=True)

            def generator():
                while True:
                    complete = result.is_complete
                    if complete:
                        break
                    yield from result.stderr

            pbar = tqdm(generator())
            for i in pbar:
                pbar.set_description(i)
    else:  # Google drive download using gdown library
        gdown.download(url, quiet=False, fuzzy=True, output=out_path)


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
    -----
        url (str): url to download the file
        filename (str): name used to store the downloaded file
        md5_hash (str): hash used to ensure the integrity of the downloaded file
        out_dir (str, optional): directory where the downloaded file will be stored. Defaults to ".".
        extract (bool, optional): Select to whether to extract the data or not. Defaults to False.
        max_trials (int, optional): Max number of trials to download the dataset. Defaults to 3.
        output (bool, optional): whether to return a list with the paths of the downloaded/extractred files. Defaults to True.

    Raises:
    -------
        ValueError: Raise an error if it fails downloading the dataset or the given md5 hash is not correct.
    """
    full_path = os.path.join(out_dir, filename)
    check_dir_exists(out_dir)
    i = 0
    while not (check_file_exists(full_path) and check_integrity(full_path, md5_hash)):
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


def check_min_arguments(func: Callable, min_args: int = 1):
    """Function that inspect the minumum number of arguments of a given function.

    Args:
    -----
        func (Callable): Function to inspect
        min_args (int, optional): Minimum number of arguments that the function
        func must have. Defaults to 1.

    Raises:
    -------
        AssertionError: Raise an assertion error if the number of arguments of the
        given function is lower than the min_args value.
    """
    signature = inspect.signature(func)
    return len(signature.parameters) >= min_args
