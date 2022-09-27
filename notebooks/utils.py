"""File with some utils functions to use on the notebooks, as for a known bug of multriprocessing library
Issue on GitHub: https://github.com/ipython/ipython/issues/10894
Error and explanation on Stack Overflow: https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror/42383397
"""

from copy import deepcopy

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, itos, stoi, sequence_len):
        text = " ".join([sentence[0] for sentence in text])
        # self.text = ' '.join([tokenizer(sentence[0]) for sentence in text]))
        self.text = tokenizer(text)
        self.tokenizer = tokenizer
        self.vocab_itos = itos
        self.vocab_stoi = stoi
        self.sequence_length = sequence_len
        # self.words_indexes = [[stoi.get(word, stoi['UNK']) for word in sentence] for sentence in text]
        self.words_indexes = [stoi.get(word, stoi.get("UNK")) for word in self.text]

    def __len__(self):
        # return len(self.text)
        return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index : index + self.sequence_length]),
            torch.tensor(
                self.words_indexes[index + 1 : index + self.sequence_length + 1]
            ),
        )


def print_function(client, *args, **kwargs):
    print(f"Client's type: {type(client)}")
    print(f"Kwargs: {kwargs.keys()}")


def add_torch_dataset_to_client(client, *args, **kwargs):
    """Function to create a dataset for each client. We keep the
    X_data property as we don't want to change the raw text, but
    it should be changed for less memory usage.

    Args:
        client (FlexDataObject): Client to create a TextDataset

    Returns:
        FlexDataObject: Client with a TextDataset in her data
    """
    new_client = deepcopy(client)
    new_client_dataset = TextDataset(
        new_client.X_data,
        kwargs["tokenizer"],
        kwargs["itos"],
        kwargs["stoi"],
        kwargs["sequence_len"],
    )
    new_client.dataset = new_client_dataset
    return new_client
