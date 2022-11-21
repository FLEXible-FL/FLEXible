from abc import abstractmethod, ABC


class EncapsulatedDataset(ABC):
    def __init__(self, out_dir='.', split=None, return_test=False):
        self.__out_dir = out_dir
        self.__split = split
        self.__return_test = return_test

    @property
    def out_dir(self):
        return self.__out_dir

    @property
    def split(self):
        return self.__split

    @property
    def return_test(self):
        return self.__return_test

    @abstractmethod
    def load_dataset(self):
        pass
