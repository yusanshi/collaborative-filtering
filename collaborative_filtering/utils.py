import time
import os
import logging
import coloredlogs
import datetime
import copy

from .parameters import parse_args

args = parse_args()


def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


def create_logger():
    logger = logging.getLogger(__name__)
    coloredlogs.install(level='DEBUG',
                        logger=logger,
                        fmt='%(asctime)s %(levelname)s %(message)s')
    log_dir = os.path.join(
        args.log_path,
        f'{args.model_name}-{args.loss_type}-{get_dataset_name(args.dataset_path)}'
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(
        log_dir,
        f"{str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')}{'-remark-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}.txt"
    )
    logger.info(f'Check {log_file_path} for the log of this run')
    file_hander = logging.FileHandler(log_file_path)
    logger.addHandler(file_hander)
    return logger


def copy_arguments(f):
    def selectively_copy(x):
        if isinstance(x, list) or isinstance(x, dict):
            return copy.deepcopy(x)
        else:
            return x

    def wrapper(*args, **kwargs):
        args = tuple(selectively_copy(x) for x in args)
        kwargs = {k: selectively_copy(v) for k, v in kwargs.items()}
        return f(*args, **kwargs)

    return wrapper


@copy_arguments
def deep_apply(d, f=lambda x: f'{x:.4f}'):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = deep_apply(v, f)
        else:
            d[k] = f(v)
    return d


def dict2table(d, k_fn=str, v_fn=lambda x: f'{x:.4f}'):
    '''
    Convert a nested dict to markdown table
    '''
    def parse_header(d, depth=0):
        if isinstance(list(d.values())[0], dict):
            header = parse_header(list(d.values())[0], depth=depth + 1)
            for v in d.values():
                assert header == parse_header(v, depth=depth + 1)
            return header
        else:
            return f"| {' | '.join([''] * depth + list(map(k_fn, d.keys())))} |"

    def parse_content(d, accumulated_keys=[]):
        if isinstance(list(d.values())[0], dict):
            contents = []
            for k, v in d.items():
                contents.extend(parse_content(v, accumulated_keys + [k_fn(k)]))
            return contents
        else:
            return [
                f"| {' | '.join(accumulated_keys + list(map(v_fn, d.values())))} |"
            ]

    lines = [parse_header(d), *parse_content(d)]
    return '\n'.join(lines)


def get_dataset_name(dataset_path):
    return os.path.abspath(dataset_path).split('/')[-1]
