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
import copy
import pickle
import random
import unittest

import pytest

from flex.data.lazy_indexable import LazyIndexable

DEFAULT_LENGTH = 10
TEST_INIT_LEN_GUESS = 3
TEST_BIGGER_INIT_LEN_GUESS = 150


def get_generator(length=DEFAULT_LENGTH):
    return (x for x in range(length))


def get_iterator(length=DEFAULT_LENGTH):
    return iter(range(length))


def get_list(length=DEFAULT_LENGTH):
    return list(range(length))


def same_contents(reference, indexable: LazyIndexable):
    return all(x == y for x, y in zip(reference, indexable))


def same_contents_with_length(
    reference, indexable: LazyIndexable, length=DEFAULT_LENGTH
):
    return (
        all(x == y for x, y in zip(reference, indexable)) and len(indexable) == length
    )


def same_contects_negative_indexing(
    reference, indexable: LazyIndexable, length=DEFAULT_LENGTH
):
    return all(indexable[i - length] == x for i, x in enumerate(reference))


def same_contents_slicing(reference, indexable: LazyIndexable):
    return all(indexable[i:][0] == x for i, x in enumerate(reference))


def same_contents_anidated_slicing(reference, indexable: LazyIndexable):
    result = []
    anidated_slice = indexable
    for i, x in enumerate(reference):
        if i > 0 and i < (DEFAULT_LENGTH - 1):
            anidated_slice = anidated_slice[1:]
            result.append(anidated_slice[0] == x)
    return all(result)


def same_contects_split_in_two_slices(reference, indexable: LazyIndexable, split_index):
    result = []
    slice_1, slice_2 = indexable[:split_index], indexable[split_index:]
    for j, x in enumerate(reference):
        if j < split_index:
            result.append(slice_1[j] == x)
        else:
            result.append(slice_2[j - split_index] == x)
    return all(result)


class TestLazySliceable(unittest.TestCase):
    def test_copy_operator_from_array(self):
        base_list = get_list()
        from_array = LazyIndexable(base_list, length=DEFAULT_LENGTH)
        from_array_copy = copy.deepcopy(from_array)
        assert same_contents_with_length(from_array, from_array_copy)

    def test_copy_operator_from_iter(self):
        base_iter = get_iterator()
        from_iter = LazyIndexable(base_iter, length=DEFAULT_LENGTH)
        from_iter_copy = copy.deepcopy(from_iter)
        assert same_contents_with_length(from_iter, from_iter_copy)

    def test_copy_operator_from_generator(self):
        generator = get_generator()
        from_gen = LazyIndexable(generator, length=DEFAULT_LENGTH)
        from_gen_copy = copy.deepcopy(from_gen)
        assert same_contents_with_length(from_gen, from_gen_copy)

    def test_copy_operator_from_array_to_numpy(self):
        base_list = get_list()
        from_array = LazyIndexable(base_list, length=DEFAULT_LENGTH)
        from_array_copy = copy.deepcopy(from_array)
        assert same_contents_with_length(
            from_array.to_numpy(), from_array_copy.to_numpy()
        )

    def test_copy_operator_from_iter_to_numpy(self):
        base_iter = get_iterator()
        from_iter = LazyIndexable(base_iter, length=DEFAULT_LENGTH)
        from_iter_copy = copy.deepcopy(from_iter)
        assert same_contents_with_length(
            from_iter.to_numpy(), from_iter_copy.to_numpy()
        )

    def test_copy_operator_from_generator_to_numpy(self):
        generator = get_generator()
        from_gen = LazyIndexable(generator, length=DEFAULT_LENGTH)
        from_gen_copy = copy.deepcopy(from_gen)
        assert same_contents_with_length(from_gen.to_numpy(), from_gen_copy.to_numpy())

    def test_from_array(self):
        base_list = get_list()
        from_array = LazyIndexable(base_list, length=DEFAULT_LENGTH)
        assert same_contents(base_list, from_array)

    def test_from_generator(self):
        generator = get_generator()
        from_gen = LazyIndexable(get_generator(), length=DEFAULT_LENGTH)
        assert same_contents(generator, from_gen)
        from_gen = LazyIndexable(get_generator(), length=DEFAULT_LENGTH)
        assert same_contents(generator, from_gen)

    def test_from_iter(self):
        iterator = get_iterator()
        from_iter = LazyIndexable(get_iterator(), length=DEFAULT_LENGTH)
        assert same_contents(iterator, from_iter)
        from_iter = LazyIndexable(get_iterator(), length=DEFAULT_LENGTH)
        assert same_contents(iterator, from_iter)

    def test_negative_indexing_with_len_from_generator(self):
        generator = get_generator()
        from_gen = LazyIndexable(get_generator(), DEFAULT_LENGTH)
        assert same_contects_negative_indexing(generator, from_gen)

    def test_negative_indexing_with_len_from_iter(self):
        iterator = get_iterator()
        from_iter = LazyIndexable(get_iterator(), DEFAULT_LENGTH)
        assert same_contects_negative_indexing(iterator, from_iter)

    def test_from_array_to_numpy(self):
        base_list = get_list()
        from_array = LazyIndexable(base_list, length=DEFAULT_LENGTH)
        from_array = from_array.to_numpy()
        assert same_contents(base_list, from_array)
        base_list = get_list()
        from_array = LazyIndexable(base_list, length=DEFAULT_LENGTH)
        from_array = from_array.to_numpy()
        assert same_contents(base_list, from_array)

    def test_from_generator_to_numpy(self):
        generator = get_generator()
        from_gen = LazyIndexable(get_generator(), length=DEFAULT_LENGTH)
        from_gen = from_gen.to_numpy()
        assert same_contents(generator, from_gen)
        generator = get_generator()
        from_gen = LazyIndexable(get_generator(), length=DEFAULT_LENGTH)
        from_gen = from_gen.to_numpy()
        assert same_contents(generator, from_gen)

    def test_from_generator_with_len_to_numpy(self):
        generator = get_generator()
        from_gen = LazyIndexable(
            get_generator(),
            length=DEFAULT_LENGTH,
        )
        from_gen = from_gen.to_numpy()
        assert len(from_gen) == DEFAULT_LENGTH
        assert same_contents(generator, from_gen)
        generator = get_generator()
        from_gen = LazyIndexable(
            get_generator(),
            length=DEFAULT_LENGTH,
        )
        from_gen = from_gen.to_numpy()
        assert same_contents_with_length(generator, from_gen)

    def test_from_iter_to_numpy(self):
        iterator = get_iterator()
        from_iter = LazyIndexable(get_iterator(), length=DEFAULT_LENGTH)
        from_iter = from_iter.to_numpy()
        assert same_contents_with_length(iterator, from_iter)
        iterator = get_iterator()
        from_iter = LazyIndexable(get_iterator(), length=DEFAULT_LENGTH)
        from_iter = from_iter.to_numpy()
        assert same_contents_with_length(iterator, from_iter)

    def test_from_iter_with_len_to_numpy(self):
        iterator = get_iterator()
        from_iter = LazyIndexable(
            get_iterator(),
            length=DEFAULT_LENGTH,
        )
        from_iter = from_iter.to_numpy()
        assert same_contents_with_length(iterator, from_iter)
        iterator = get_iterator()
        from_iter = LazyIndexable(
            get_iterator(),
            length=DEFAULT_LENGTH,
        )
        from_iter = from_iter.to_numpy()
        assert same_contents_with_length(iterator, from_iter)

    def test_slicing_from_generator(self):
        generator = get_generator()
        from_gen = LazyIndexable(get_generator(), length=DEFAULT_LENGTH)
        assert same_contents_slicing(generator, from_gen)
        generator = get_generator()
        from_gen = LazyIndexable(get_generator(), length=DEFAULT_LENGTH)
        assert same_contents_slicing(generator, from_gen)

    def test_slicing_with_len_from_generator(self):
        generator = get_generator()
        from_gen = LazyIndexable(get_generator(), DEFAULT_LENGTH)
        assert same_contents_slicing(generator, from_gen)

    def test_slicing_with_len_from_iterator(self):
        iterator = get_iterator()
        from_iter = LazyIndexable(get_iterator(), DEFAULT_LENGTH)
        assert same_contents_slicing(iterator, from_iter)

    def test_slicing_from_iterator(self):
        iterator = get_iterator()
        from_iter = LazyIndexable(get_iterator(), length=DEFAULT_LENGTH)
        assert same_contents_slicing(iterator, from_iter)
        iterator = get_iterator()
        from_iter = LazyIndexable(get_iterator(), length=DEFAULT_LENGTH)
        assert same_contents_slicing(iterator, from_iter)

    def test_anidated_slices_with_len_from_generator(self):
        generator = get_generator()
        from_gen = LazyIndexable(get_generator(), DEFAULT_LENGTH)
        assert same_contents_anidated_slicing(generator, from_gen)

    def test_anidated_slices_from_generator(self):
        generator = get_generator()
        from_gen = LazyIndexable(get_generator(), length=DEFAULT_LENGTH)
        assert same_contents_anidated_slicing(generator, from_gen)

    def test_anidated_slices_with_len_from_iterator(self):
        iterator = get_iterator()
        from_iter = LazyIndexable(get_iterator(), DEFAULT_LENGTH)
        assert same_contents_anidated_slicing(iterator, from_iter)

    def test_anidated_slices_from_iterator(self):
        iterator = get_iterator()
        from_iter = LazyIndexable(get_iterator(), length=DEFAULT_LENGTH)
        assert same_contents_anidated_slicing(iterator, from_iter)

    def test_complex_slicing_with_len_from_generator(self):
        for i in range(DEFAULT_LENGTH):
            generator = get_generator()
            from_gen = LazyIndexable(get_generator(), DEFAULT_LENGTH)
            assert same_contects_split_in_two_slices(
                generator, from_gen, i
            ), f"Error at iteration: {i}"

    def test_complex_slicing_from_generator(self):
        for i in range(DEFAULT_LENGTH):
            generator = get_generator()
            from_gen = LazyIndexable(get_generator(), length=DEFAULT_LENGTH)
            assert same_contects_split_in_two_slices(
                generator, from_gen, i
            ), f"Error at iteration: {i}"

    def test_complex_slicing_with_len_from_iterator(self):
        for i in range(DEFAULT_LENGTH):
            gen_iter = LazyIndexable(get_iterator(), DEFAULT_LENGTH)
            iterator = get_iterator()
            assert same_contects_split_in_two_slices(
                iterator, gen_iter, i
            ), f"Error at iteration: {i}"

    def test_complex_slicing_from_iterator(self):
        for i in range(DEFAULT_LENGTH):
            gen_iter = LazyIndexable(get_iterator(), length=DEFAULT_LENGTH)
            iterator = get_iterator()
            assert same_contects_split_in_two_slices(
                iterator, gen_iter, i
            ), f"Error at iteration: {i}"

    def test_random_indexing_with_len_from_iterator(self):
        length = 100
        sample_size = 10
        random.seed(sample_size)
        available_indexes = get_list(length)
        for i in range(50):
            selected_indexes = random.sample(available_indexes, sample_size)
            from_iter = LazyIndexable(get_iterator(length=length), length)
            assert same_contents_with_length(
                selected_indexes, from_iter[selected_indexes], length=sample_size
            ), f"Error at iteration: {i}"

    def test_random_negative_indexing_with_len_from_iterator(self):
        length = 100
        sample_size = 10
        random.seed(sample_size)
        available_indexes = get_list(length)
        for i in range(50):
            selected_indexes = random.sample(available_indexes[1:], sample_size)
            negative_indexes = [-i for i in selected_indexes]
            positive_equivalent = [i + length for i in negative_indexes]
            from_iter = LazyIndexable(get_iterator(length=length), length)
            assert same_contents_with_length(
                positive_equivalent, from_iter[negative_indexes], length=sample_size
            ), f"Error at iteration: {i}"

    def test_random_indexing_from_iterator(self):
        length = 100
        sample_size = 10
        random.seed(sample_size)
        available_indexes = get_list(length)
        for i in range(50):
            selected_indexes = random.sample(available_indexes, sample_size)
            from_iter = LazyIndexable(
                get_iterator(length=length),
                length=length,
            )
            assert same_contents(
                selected_indexes, from_iter[selected_indexes]
            ), f"Error at iteration: {i}"
        for i in range(50):
            selected_indexes = random.sample(available_indexes, sample_size)
            from_iter = LazyIndexable(
                get_iterator(length=length),
                length=length,
            )
            assert same_contents(
                selected_indexes, from_iter[selected_indexes]
            ), f"Error at iteration: {i}"

    def test_random_indexing_with_len_from_generator(self):
        length = 100
        sample_size = 10
        random.seed(sample_size)
        available_indexes = get_list(length)
        for i in range(50):
            selected_indexes = random.sample(available_indexes, sample_size)
            from_gen = LazyIndexable(get_generator(length=length), length)
            assert same_contents_with_length(
                selected_indexes, from_gen[selected_indexes], length=sample_size
            ), f"Error at iteration: {i}"

    def test_random_negative_indexing_with_len_from_generator(self):
        length = 100
        sample_size = 10
        random.seed(sample_size)
        available_indexes = get_list(length)
        for j in range(50):
            selected_indexes = random.sample(available_indexes[1:], sample_size)
            negative_indexes = [-i for i in selected_indexes]
            positive_equivalent = [i + length for i in negative_indexes]
            from_gen = LazyIndexable(get_generator(length=length), length)
            assert same_contents_with_length(
                positive_equivalent, from_gen[negative_indexes], length=sample_size
            ), f"Error at iteration: {j}"

    def test_random_indexing_from_generator(self):
        length = 100
        sample_size = 10
        random.seed(sample_size)
        available_indexes = get_list(length)
        for i in range(50):
            selected_indexes = random.sample(available_indexes, sample_size)
            from_gen = LazyIndexable(
                get_generator(length=length),
                length=length,
            )
            assert same_contents(
                selected_indexes, from_gen[selected_indexes]
            ), f"Error at iteration: {i}"
        for i in range(50):
            selected_indexes = random.sample(available_indexes, sample_size)
            from_gen = LazyIndexable(
                get_generator(length=length),
                length=length,
            )
            assert same_contents(
                selected_indexes, from_gen[selected_indexes]
            ), f"Error at iteration: {i}"

    def test_random_negative_indexing_with_len_from_iterator_to_numpy(self):
        length = 100
        sample_size = 10
        random.seed(sample_size)
        available_indexes = get_list(length)
        for i in range(50):
            selected_indexes = random.sample(available_indexes[1:], sample_size)
            negative_indexes = [-i for i in selected_indexes]
            positive_equivalent = [i + length for i in negative_indexes]
            from_iter = LazyIndexable(get_iterator(length=length), length)
            sampled_indexes = from_iter[negative_indexes].to_numpy()
            assert same_contents_with_length(
                positive_equivalent, sampled_indexes, length=sample_size
            ), f"Error at iteration: {i}"

    def test_random_indexing_from_iterator_to_numpy(self):
        length = 100
        sample_size = 10
        random.seed(sample_size)
        available_indexes = get_list(length)
        for i in range(50):
            selected_indexes = random.sample(available_indexes, sample_size)
            from_iter = LazyIndexable(get_iterator(length=length), length=length)
            assert same_contents(
                selected_indexes, from_iter[selected_indexes].to_numpy()
            ), f"Error at iteration: {i}"
        for i in range(50):
            selected_indexes = random.sample(available_indexes, sample_size)
            from_iter = LazyIndexable(
                get_iterator(length=length),
                length=length,
            )
            assert same_contents(
                selected_indexes, from_iter[selected_indexes].to_numpy()
            ), f"Error at iteration: {i}"

    def test_random_indexing_with_len_from_generator_to_numpy(self):
        length = 1000
        sample_size = length // 2
        trials = 100
        random.seed(sample_size)
        available_indexes = get_list(length)
        for i in range(trials):
            selected_indexes = random.sample(available_indexes, sample_size)
            from_gen = LazyIndexable(get_generator(length=length), length)
            assert same_contents_with_length(
                selected_indexes,
                from_gen[selected_indexes].to_numpy(),
                length=sample_size,
            ), f"Error at iteration: {i}"

    def test_random_negative_indexing_with_len_from_generator_to_numpy(self):
        length = 1000
        sample_size = 10
        trials = 100
        random.seed(sample_size)
        available_indexes = get_list(length)
        for i in range(trials):
            selected_indexes = random.sample(available_indexes[1:], sample_size)
            negative_indexes = [-i for i in selected_indexes]
            positive_equivalent = [i + length for i in negative_indexes]
            from_gen = LazyIndexable(get_generator(length=length), length)
            assert same_contents_with_length(
                positive_equivalent,
                from_gen[negative_indexes].to_numpy(),
                length=sample_size,
            ), f"Error at iteration: {i}"

    def test_random_indexing_from_generator_to_numpy(self):
        length = 100
        sample_size = 10
        trials = 100
        random.seed(sample_size)
        available_indexes = get_list(length)
        for i in range(trials):
            selected_indexes = random.sample(available_indexes, sample_size)
            from_gen = LazyIndexable(get_generator(length=length), length=length)
            assert same_contents(
                selected_indexes, from_gen[selected_indexes].to_numpy()
            ), f"Error at iteration: {i}"
        for i in range(trials):
            selected_indexes = random.sample(available_indexes, sample_size)
            from_gen = LazyIndexable(
                get_generator(length=length),
                length=length,
            )
            assert same_contents(
                selected_indexes, from_gen[selected_indexes].to_numpy()
            ), f"Error at iteration: {i}"

    def test_empty_slice_with_len_from_generator(self):
        from_gen = LazyIndexable(get_generator(), DEFAULT_LENGTH)[:0]
        with pytest.raises(IndexError):
            from_gen[0]
        from_gen = LazyIndexable(get_generator(), DEFAULT_LENGTH)[
            5:2
        ]  # start greater than stop
        with pytest.raises(IndexError):
            from_gen[0]

    def test_empty_slice_from_generator(self):
        from_gen = LazyIndexable(get_generator(), length=DEFAULT_LENGTH)[:0]
        with pytest.raises(IndexError):
            from_gen[0]
        from_gen = LazyIndexable(get_generator(), length=DEFAULT_LENGTH)[
            5:2
        ]  # start greater than stop
        with pytest.raises(IndexError):
            from_gen[0]

    def test_empty_slice_with_len_from_iterator(self):
        from_iter = LazyIndexable(get_iterator(), DEFAULT_LENGTH)
        with pytest.raises(IndexError):
            from_iter[:0][0]
        from_iter = LazyIndexable(get_iterator(), DEFAULT_LENGTH)[
            5:2
        ]  # start greater than stop
        with pytest.raises(IndexError):
            from_iter[0]

    def test_empty_slice_from_iterator(self):
        from_iter = LazyIndexable(get_iterator(), length=DEFAULT_LENGTH)
        with pytest.raises(IndexError):
            from_iter[:0][0]
        from_iter = LazyIndexable(get_iterator(), length=DEFAULT_LENGTH)[
            5:2
        ]  # start greater than stop
        with pytest.raises(IndexError):
            from_iter[0]

    def test_index_out_of_bounds_with_len_from_generator(self):
        from_gen = LazyIndexable(get_generator(), DEFAULT_LENGTH)
        with pytest.raises(IndexError):
            from_gen[DEFAULT_LENGTH * DEFAULT_LENGTH]

    def test_index_out_of_bounds_from_generator(self):
        from_gen = LazyIndexable(get_generator(), length=DEFAULT_LENGTH)
        with pytest.raises(IndexError):
            from_gen[DEFAULT_LENGTH * DEFAULT_LENGTH]

    def test_index_out_of_bounds_with_len_from_iterator(self):
        from_iter = LazyIndexable(get_iterator(), DEFAULT_LENGTH)
        with pytest.raises(IndexError):
            from_iter[DEFAULT_LENGTH * DEFAULT_LENGTH]

    def test_pickle_from_array(self):
        base_list = get_list()
        from_array = LazyIndexable(base_list, length=DEFAULT_LENGTH)
        from_array_pickle_copy = pickle.loads(pickle.dumps(from_array))
        assert same_contents_with_length(from_array, from_array_pickle_copy)

    def test_pickle_from_iter(self):
        base_iter = get_iterator()
        from_iter = LazyIndexable(base_iter, length=DEFAULT_LENGTH)
        from_iter_pickle_copy = pickle.loads(pickle.dumps(from_iter))
        assert same_contents_with_length(from_iter, from_iter_pickle_copy)

    def test_pickle_from_generator(self):
        base_generator = get_generator()
        from_generator = LazyIndexable(base_generator, length=DEFAULT_LENGTH)
        from_generator_pickle_copy = pickle.loads(pickle.dumps(from_generator))
        assert same_contents_with_length(from_generator, from_generator_pickle_copy)
