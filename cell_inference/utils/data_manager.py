import os
import numpy as np

KEY_TYPE = {'x': 'array', 'y': 'array', 'ys': 'array',
            't': 'unique', 'rand_param': 'array', 'gmax': 'array',
            'bad_indices': 'dict_index', 'good_indices': 'index', 'invalid_params': 'dict_array'}

class NpzFilesCollector(object):
    def __init__(self, filename):
        self.closed = True
        self.__get_files(filename)
        self.__load()

    # PRIVATE METHODS
    def __get_files(self, filename):
        path, filename = os.path.split(filename)
        fname, fext = os.path.splitext(filename)
        file_names = [f for f in os.listdir(path) if f.startswith(fname) and f.endswith(fext)]
        if not file_names:
            raise FileNotFoundError('No file match pattern %s*%s' % (fname, fext))
        batch_id = list(map(lambda f: int(''.join(filter(str.isdigit, f))), file_names))
        self.batch_id, file_names = zip(*sorted(zip(batch_id, file_names)))
        self.file_names = [os.path.join(path, f) for f in file_names]
        self.nfile = len(self.file_names)

    def __load(self):
        self.mem_maps = [np.load(f, mmap_mode='r', allow_pickle=True) for f in self.file_names]
        self.closed = False

    def __getitem__(self, key):
        if key in self.mem_maps[0]:
            fcn = getattr(self, KEY_TYPE.get(key, 'noattr'))
        else:
            fcn = self.noattr
        return fcn(key=key)

    def __enter__(self):
        if self.closed:
            self.__load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for f in self.mem_maps:
            f.close()
        self.closed = True

    # PUBLIC METHODS
    def noattr(self, key):
        raise ValueError('Attribute "%s" does not exist.' % (key))

    def null(self, key):
        raise ValueError('Attribute "%s" is not implemented.' % (key))
        
    def _key_to_array(fcn):
        def wrapper(self, key=None, list_=None):
            if list_ is None:
                list_ = [mmap[key] for mmap in self.mem_maps]
            return fcn(self, list_)
        return wrapper

    def unique(self, key):
        return self.mem_maps[0][key]

    @_key_to_array
    def array(self, list_):
        list_0 = np.asarray(list_[0])
        if list_0.ndim == 0:
            value = list_0.item()
        else:
            value =  np.concatenate(list_, axis=0)
        return value

    def get_number_samples(self, total=True):
        key = 'nsample_total' if total else 'nsample_good'
        if hasattr(self, key):
            value = getattr(self, key)
        else:
            if total:
                value = [sum(map(len, mmap['bad_indices'].item().values())) for mmap in self.mem_maps]
            else:
                value = [len(mmap['x']) for mmap in self.mem_maps]
            setattr(self, key, value)
        return value

    def get_init_idx(self):
        if not hasattr(self, 'init_idx'):
            self.init_idx = np.cumsum(np.insert(self.get_number_samples()[:-1], 0, 0))
        return self.init_idx

    @_key_to_array
    def index(self, list_):
        return np.concatenate([(i + np.asarray(idx)).astype(int) for i, idx in zip(self.get_init_idx(), list_)])

    def to_dict_list(self, key):
        dicts = [mmap[key].item() for mmap in self.mem_maps]
        return {key: [d[key] for d in dicts] for key in dicts[0]}

    def dict_index(self, key):
        dict_indices = self.to_dict_list(key)
        return {key: self.index(list_=value) for key, value in dict_indices.items()}

    def dict_array(self, key):
        dict_arrays = self.to_dict_list(key)
        return {key: self.array(list_=value) for key, value in dict_arrays.items()}
