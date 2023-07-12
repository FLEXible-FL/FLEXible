import random
import unittest
import pytest

from flex.data.lazy_indexable import LazyIndexable

DEFAULT_LENGTH = 10


def get_generator(length=DEFAULT_LENGTH):
    return (x for x in range(length))


def get_iterator(length=DEFAULT_LENGTH):
    return iter(range(length))


def get_list(length=DEFAULT_LENGTH):
    return list(range(length))


def same_contents(reference, sliceable: LazyIndexable):
    return all(x == y for x, y in zip(reference, sliceable))


def same_contects_negative_indexing(reference, sliceable: LazyIndexable):
    return all(sliceable[i - len(sliceable)] == x for i, x in enumerate(reference))


def same_contents_slicing(reference, sliceable: LazyIndexable):
    return all(sliceable[i:][0] == x for i, x in enumerate(reference))


def same_contents_anidated_slicing(reference, sliceable: LazyIndexable):
    result = []
    anidated_slice = sliceable
    for i, x in enumerate(reference):
        if i > 0 and i < (len(sliceable) - 1):
            anidated_slice = anidated_slice[1:]
            result.append(anidated_slice[0] == x)
    return all(result)


def same_contects_split_in_two_slices(reference, sliceable: LazyIndexable, split_index):
    result = []
    slice_1, slice_2 = sliceable[:split_index], sliceable[split_index:]
    for j, x in enumerate(reference):
        if j < split_index:
            result.append(slice_1[j] == x)
        else:
            result.append(slice_2[j - split_index] == x)
    return all(result)


class TestLazySliceable(unittest.TestCase):
    def test_from_list(self):
        base_list = get_list()
        from_list = LazyIndexable(base_list, DEFAULT_LENGTH)
        assert same_contents(base_list, from_list)

    def test_from_generator(self):
        generator = get_generator()
        from_gen = LazyIndexable(get_generator(), DEFAULT_LENGTH)
        assert same_contents(generator, from_gen)

    def test_from_iter(self):
        iterator = get_iterator()
        from_iter = LazyIndexable(get_iterator(), DEFAULT_LENGTH)
        assert same_contents(iterator, from_iter)

    def test_negative_indexing_from_generator(self):
        generator = get_generator()
        from_gen = LazyIndexable(get_generator(), DEFAULT_LENGTH)
        assert same_contects_negative_indexing(generator, from_gen)

    def test_negative_indexing_from_iter(self):
        iterator = get_iterator()
        from_iter = LazyIndexable(get_iterator(), DEFAULT_LENGTH)
        assert same_contects_negative_indexing(iterator, from_iter)

    def test_from_list_to_numpy(self):
        base_list = get_list()
        from_list = LazyIndexable(base_list, DEFAULT_LENGTH).to_numpy()
        assert same_contents(base_list, from_list)

    def test_from_generator_to_numpy(self):
        generator = get_generator()
        from_gen = LazyIndexable(get_generator(), DEFAULT_LENGTH).to_numpy()
        assert same_contents(generator, from_gen)

    def test_from_iter_to_numpy(self):
        iterator = get_iterator()
        from_iter = LazyIndexable(get_iterator(), DEFAULT_LENGTH).to_numpy()
        assert same_contents(iterator, from_iter)

    def test_slicing_from_generator(self):
        generator = get_generator()
        from_gen = LazyIndexable(get_generator(), DEFAULT_LENGTH)
        assert same_contents_slicing(generator, from_gen)

    def test_slicing_from_iterator(self):
        iterator = get_iterator()
        from_iter = LazyIndexable(get_iterator(), DEFAULT_LENGTH)
        assert same_contents_slicing(iterator, from_iter)

    def test_anidated_slices_from_generator(self):
        generator = get_generator()
        from_gen = LazyIndexable(get_generator(), DEFAULT_LENGTH)
        assert same_contents_anidated_slicing(generator, from_gen)

    def test_anidated_slices_from_iterator(self):
        iterator = get_iterator()
        from_iter = LazyIndexable(get_iterator(), DEFAULT_LENGTH)
        assert same_contents_anidated_slicing(iterator, from_iter)

    def test_complex_slicing_from_generator(self):
        for i in range(DEFAULT_LENGTH):
            iterator = get_iterator()
            from_iter = LazyIndexable(get_iterator(), DEFAULT_LENGTH)
            assert same_contects_split_in_two_slices(iterator, from_iter, i)

    def test_complex_slicing_from_iterator(self):
        for i in range(DEFAULT_LENGTH):
            from_gen = LazyIndexable(get_generator(), DEFAULT_LENGTH)
            generator = get_generator()
            assert same_contects_split_in_two_slices(generator, from_gen, i)

    def test_random_indexing_from_iterator(self):
        length = 100
        sample_size = 10
        random.seed(sample_size)
        available_indexes = get_list(length)
        for _ in range(50):
            selected_indexes = random.sample(available_indexes, sample_size)
            from_iter = LazyIndexable(get_iterator(length=length), length)
            assert same_contents(selected_indexes, from_iter[selected_indexes])

    def test_random_indexing_from_generator(self):
        length = 100
        sample_size = 10
        random.seed(sample_size)
        available_indexes = get_list(length)
        for _ in range(50):
            selected_indexes = random.sample(available_indexes, sample_size)
            from_gen = LazyIndexable(get_generator(length=length), length)
            assert same_contents(selected_indexes, from_gen[selected_indexes])

    def test_random_indexing_from_iterator_to_numpy(self):
        length = 100
        sample_size = 10
        random.seed(sample_size)
        available_indexes = get_list(length)
        for _ in range(50):
            selected_indexes = random.sample(available_indexes, sample_size)
            from_iter = LazyIndexable(get_iterator(length=length), length)
            assert same_contents(
                selected_indexes, from_iter[selected_indexes].to_numpy()
            )

    def test_random_indexing_from_generator_to_numpy(self):
        length = 1000
        sample_size = 10
        trials = 100
        random.seed(sample_size)
        available_indexes = get_list(length)
        for _ in range(trials):
            selected_indexes = random.sample(available_indexes, sample_size)
            from_gen = LazyIndexable(get_generator(length=length), length)
            assert same_contents(
                selected_indexes, from_gen[selected_indexes].to_numpy()
            )

    def test_empty_slice_from_generator(self):
        from_gen = LazyIndexable(get_generator(), DEFAULT_LENGTH)[:0]
        with pytest.raises(IndexError):
            from_gen[0]

    def test_empty_slice_from_iterator(self):
        from_iter = LazyIndexable(get_iterator(), DEFAULT_LENGTH)
        with pytest.raises(IndexError):
            from_iter[:0][0]

    def test_index_out_of_bounds_from_generator(self):
        from_gen = LazyIndexable(get_generator(), DEFAULT_LENGTH)
        with pytest.raises(IndexError):
            from_gen[DEFAULT_LENGTH*DEFAULT_LENGTH]

    def test_index_out_of_bounds_from_iterator(self):
        from_iter = LazyIndexable(get_iterator(), DEFAULT_LENGTH)
        with pytest.raises(IndexError):
            from_iter[DEFAULT_LENGTH*DEFAULT_LENGTH]
