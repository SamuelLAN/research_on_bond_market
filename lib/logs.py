import os
import time
from lib import utils

LOG = []

MODULE = 'default'
PROCESS = 'default'

LEVEL_MSG = 'MSG'
LEVEL_NOTICE = 'NOTICE'
LEVEL_WARNING = 'WARNING'
LEVEL_ERROR = 'ERROR'

MAX_FILE_SIZE = 10 * 1024 * 1024


def add(_id, function, message, pre_sep='', empty_line=0, _level=LEVEL_MSG, show=True):
    # construct log message
    _time = str(time.strftime('%Y-%m-%d %H:%M:%S'))
    string = '\n' * empty_line + (pre_sep + '\n' if pre_sep else pre_sep) + f'{_id} : {_level} : {_time} : {function} : {message}\n'
    if show:
        print(string[:-1])

    # get correct file path
    dir_path = utils.get_relative_dir('log', MODULE, PROCESS)

    file_no = max(len(os.listdir(dir_path)), 1)
    file_path = os.path.join(dir_path, f'{file_no}.log')

    while os.path.exists(file_path) and os.path.getsize(file_path) > MAX_FILE_SIZE:
        file_no += 1
        file_path = os.path.join(dir_path, f'{file_no}.log')

    # write log
    with open(file_path, 'ab') as f:
        f.write(string.encode('utf-8'))
