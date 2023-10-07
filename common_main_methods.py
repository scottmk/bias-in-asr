import contextlib
import os
import re
import sys

from tqdm.contrib import DummyTqdmFile


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


def get_input_filepaths(input_dir_path, acceptable_exts=None):
    ext_regexp = ""
    if acceptable_exts:
        if isinstance(acceptable_exts, str):
            acceptable_exts = [acceptable_exts]
        ext_regexp = f"^.*\\.({'|'.join(acceptable_exts)})$"
    for dirpath, dirnames, filenames in os.walk(input_dir_path):
        for filename in filenames:
            if not acceptable_exts or re.match(ext_regexp, filename, re.IGNORECASE):
                if dirpath.endswith('/'):
                    dirpath = dirpath[:-1]
                yield f"{dirpath}/{filename}"
