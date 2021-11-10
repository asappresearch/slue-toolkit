"""
Compilation of commonly used functions
"""

import pickle as pkl


def save_pkl(fname, dict_name):
    with open(fname, "wb") as f:
        pkl.dump(dict_name, f)


def load_pkl(fname, encdng=None):
    if encdng is None:
        with open(fname, "rb") as f:
            data = pkl.load(f)
    else:
        with open(fname, "rb") as f:
            data = pkl.load(f, encoding=encdng)
    return data


def read_lst(fname):
    with open(fname, "r") as f:
        lst_from_file = [line.strip() for line in f.readlines()]
    return lst_from_file


def write_to_file(write_str, fname):
    with open(fname, "w") as f:
        f.write(write_str)
