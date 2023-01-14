import os
import numpy as np

KEY_TYPE = {'x': 'array', 'y': 'array', 'ys': 'array', 'rand_param': 'array', 'gmax': 'array',
            't': 'unique', 'bad_indices': 'dict_index', 'good_indices': 'index', 'firing_rate': 'array',
            'invalid_params': 'dict_array', 'valid': 'index_total', 'invalid': 'index_total'}
# Use 'null' for 'not implemented'.

class NpzFilesCollector(object):
    def __init__(self, filename):
        self.closed = True
        self.filename = filename
        self.__get_files()
        self.__load()

    # PRIVATE METHODS
    def __get_files(self):
        path, filename = os.path.split(self.filename)
        fname, fext = os.path.splitext(filename)
        n_n, n_e = len(fname), len(fext)
        # find files with number at the end of file name
        file_names = [f for f in os.listdir(path) if f.startswith(fname) and f.endswith(fext) and len(f) > n_n + n_e]
        file_names = list(filter(lambda f: any(c.isdigit() for c in f[n_n:-n_e]), file_names))
        if not file_names:
            raise FileNotFoundError('No file match pattern %s*%s' % (fname, fext))
        batch_id = list(map(lambda f: int(''.join(filter(str.isdigit, f[n_n:-n_e]))), file_names))
        self.batch_id, file_names = zip(*sorted(zip(batch_id, file_names)))
        self.file_names = [os.path.join(path, f) for f in file_names]
        self.nfile = len(self.file_names)

    def __load(self):
        self.mem_maps = [np.load(f, mmap_mode='r', allow_pickle=True) for f in self.file_names]
        self.closed = False

    def __getitem__(self, key):
        if key in self.mem_maps[0]:
            fcn = getattr(self, KEY_TYPE.get(key, 'null'))
        else:
            fcn = self.noattr
        return fcn(key=key)

    def __enter__(self):
        if self.closed:
            self.__load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # PUBLIC METHODS
    def noattr(self, key):
        raise ValueError('Attribute "%s" does not exist.' % (key))

    def null(self, key):
        raise ValueError('Attribute "%s" is not implemented.' % (key))

    def unique(self, key):
        return self.mem_maps[0][key]

    def _key_to_array(fcn):
        def wrapper(self, key=None, list_=None, **kwargs):
            if list_ is None:
                list_ = [mmap[key] for mmap in self.mem_maps]
            return fcn(self, list_, **kwargs)
        return wrapper

    @_key_to_array
    def array(self, list_):
        list_0 = np.asarray(list_[0])
        if list_0.ndim == 0:
            value = list_0.item()
        else:
            value =  np.concatenate(list_, axis=0)
        return value

    def sample_type(self, stype):
        return stype if stype in ('total', 'valid') else 'good'

    def get_number_samples(self, stype='valid'):
        key = 'nsample_' + self.sample_type(stype)
        if hasattr(self, key):
            value = getattr(self, key)
        else:
            if stype == 'valid':
                value = [sum(map(len, mmap['bad_indices'].item().values())) for mmap in self.mem_maps]
            elif stype == 'total':
                value = [len(mmap['valid']) + len(mmap['invalid']) for mmap in self.mem_maps]
            else:
                value = [len(mmap['x']) for mmap in self.mem_maps]
            setattr(self, key, value)
        return value

    def get_init_idx(self, stype='valid'):
        key = 'init_idx_' + self.sample_type(stype)
        if hasattr(self, key):
            value = getattr(self, key)
        else:
            value = np.cumsum(np.insert(self.get_number_samples(stype)[:-1], 0, 0))
            setattr(self, key, value)
        return value

    @_key_to_array
    def index(self, list_, stype='valid'):
        return np.concatenate([(i + np.asarray(idx)).astype(int) for i, idx in zip(self.get_init_idx(stype), list_)])

    def index_total(self, key):
        return self.index(key=key, stype='total')

    def to_dict_list(self, key):
        dicts = [mmap[key].item() for mmap in self.mem_maps]
        return {key: [d[key] for d in dicts] for key in dicts[0]}

    def dict_index(self, key):
        dict_indices = self.to_dict_list(key)
        return {key: self.index(list_=value, stype='valid') for key, value in dict_indices.items()}

    def dict_array(self, key):
        dict_arrays = self.to_dict_list(key)
        return {key: self.array(list_=value) for key, value in dict_arrays.items()}

    def save_to_single_file(self, filename=None):
        if filename is None:
            filename = self.filename
        output = {}
        for key in self.mem_maps[0]:
            key_type = KEY_TYPE.get(key, 'null')
            if key_type != 'null':
                output[key] = getattr(self, key_type)(key=key)
        np.savez(self.filename, **output)

    def close(self):
        if not self.closed:
            for f in self.mem_maps:
                f.close()
            self.mem_maps.clear()
            self.closed = True

    def delete_files(self):
        self.close()
        for f in self.file_names:
            os.remove(f)
