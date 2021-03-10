import os
import time
import json
import hashlib
import chardet
from six.moves import cPickle as pickle
from config.path import DATA_ROOT_DIR

# __cur_path = os.path.split(os.path.abspath(__file__))[0]
# root_dir = os.path.split(__cur_path)[0]
root_dir = DATA_ROOT_DIR


def date_2_timestamp(_date, close_night=False):
    """ convert date string (%Y-%m-%d) to timestamp """
    _time = ' 00:00:00' if not close_night else ' 23:00:00'
    return time.mktime(time.strptime(_date + _time, '%Y-%m-%d %H:%M:%S'))


def date_2_weekday(_date):
    """ convert date string (%Y-%m-%d) to weekday """
    return time.strftime('%w', time.strptime(_date, '%Y-%m-%d'))


def timestamp_2_date(timestamp):
    """ convert timestamp to date string (%Y-%m-%d) """
    return time.strftime('%Y-%m-%d', time.localtime(timestamp))


def list_2_dict(_list):
    """ convert list to dict """
    return {v: True for v in _list}


def decode_2_utf8(string):
    """ decode string to utf-8 """
    if isinstance(string, str):
        return string
    if isinstance(string, int) or isinstance(string, float):
        return str(string)
    if not isinstance(string, bytes):
        return string

    try:
        return string.decode('utf-8')
    except:
        encoding = chardet.detect(string)['encoding']
        if encoding:
            try:
                return string.decode(encoding)
            except:
                pass
        return string


def get_relative_dir(*args, root=''):
    """ return the relative path based on the root_dir; if not exists, the dir would be created """
    dir_path = root_dir if not root else root
    for arg in args:
        dir_path = os.path.join(dir_path, arg)
        if not os.path.exists(dir_path) and '.' not in arg:
            os.mkdir(dir_path)
    return dir_path


def get_relative_file(*args, root=''):
    """ return the relative path of the file based on the root_dir """
    return os.path.join(get_relative_dir(*args[:-1], root=root), args[-1])


def load_pkl(_path):
    with open(_path, 'rb') as f:
        return pickle.load(f)


def write_pkl(_path, data):
    with open(_path, 'wb') as f:
        pickle.dump(data, f)


def load_json(_path):
    with open(_path, 'rb') as f:
        return json.load(f)


def write_json(_path, data):
    with open(_path, 'wb') as f:
        f.write(json.dumps(data).encode('utf-8'))


def cache(file_name, data):
    """ cache data in the root_dir/runtime/cache """
    file_path = os.path.join(get_relative_dir('runtime', 'cache'), file_name)
    write_pkl(file_path, data)


def read_cache(file_name):
    """ read data from cache in the root_dir/runtime/cache """
    file_path = os.path.join(get_relative_dir('runtime', 'cache'), file_name)
    if not os.path.exists(file_path):
        return
    return load_pkl(file_path)
