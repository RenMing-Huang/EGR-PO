from typing import Any
import numpy as np


def get_size(data):
    if isinstance(data, dict):
        sizes = [get_size(v) for v in data.values()]
        return max(sizes)
    elif isinstance(data, np.ndarray):
        return len(data)
    else:
        raise ValueError("Unsupported data type")


class Dataset(object):
    """
    A class for storing (and retrieving batches of) data in nested dictionary format.

    Example:
        dataset = Dataset({
            'observations': {
                'image': np.random.randn(100, 28, 28, 1),
                'state': np.random.randn(100, 4),
            },
            'actions': np.random.randn(100, 2),
        })

        batch = dataset.sample(32)
        # Batch should have nested shape: {
        # 'observations': {'image': (32, 28, 28, 1), 'state': (32, 4)},
        # 'actions': (32, 2)
        # }
    """
    def __init__(self, data):
        self.data = data
        print("keys: ", data.keys())
        self.size = get_size(data)

    @classmethod
    def create(
        cls,
        observations,
        actions,
        rewards,
        masks,
        next_observations,
        freeze=False,
        **extra_fields
    ):
        data = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "masks": masks,
            "next_observations": next_observations,
            **extra_fields,
        }
        # Force freeze
        if freeze:
            data = freeze_dict(data)
        
        return cls(data)

    def copy(self, _dict):
        for key in _dict.keys():
            if isinstance(_dict[key], dict):
                self.copy(_dict[key])
            else:
                self.data[key] = _dict[key].copy()
        return self

    def sample(self, batch_size, indx=None):
        """
        Sample a batch of data from the dataset. Use `indx` to specify a specific
        set of indices to retrieve. Otherwise, a random sample will be drawn.

        Returns a dictionary with the same structure as the original dataset.
        """
        if indx is None:
            indx = np.random.randint(self.size, size=batch_size)
        return self.get_subset(indx)


    def get_subset(self, indx):
        return tree_map(lambda arr: arr[indx], self.data)

    def __getitem__(self, key):
        """ Returns the dictionary of the dataset
        """
        return self.data[key]
    


class ReplayBuffer(Dataset):
    """
    Dataset where data is added to the buffer.

    Example:
        example_transition = {
            'observations': {
                'image': np.random.randn(28, 28, 1),
                'state': np.random.randn(4),
            },
            'actions': np.random.randn(2),
        }
        buffer = ReplayBuffer.create(example_transition, size=1000)
        buffer.add_transition(example_transition)
        batch = buffer.sample(32)

    """

    @classmethod
    def create(cls, transition, size):
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = get_subset_dict(transition, slice(None))
        buffer_dict = tree_map(create_buffer, buffer_dict)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        tree_map(set_idx, self, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)


def get_subset_dict(data, indx):
    if isinstance(data, dict):
        return {k: get_subset_dict(v, indx) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        return data[indx]
    else:
        raise ValueError("Unsupported data type")


def tree_map(fn, tree):
    if isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(fn, v) for v in tree)
    else:
        return fn(tree)


def freeze_dict(data):
    if isinstance(data, dict):
        return {k: freeze_dict(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(freeze_dict(v) for v in data)
    elif isinstance(data, np.ndarray):
        data.setflags(write=False)
        return data
    else:
        return data
    
def unfreeze_dict(data):
    if isinstance(data, dict):
        return {k: unfreeze_dict(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(unfreeze_dict(v) for v in data)
    elif isinstance(data, np.ndarray):
        data.setflags(write=True)
        return data
    else:
        return data
