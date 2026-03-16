from .core import *
from . import config


def _optional_import(module_name):
    try:
        module = __import__(f"{__name__}.{module_name}", fromlist=[module_name])
        globals()[module_name] = module
    except ModuleNotFoundError:
        # Allow lightweight submodules (e.g. custom_reviewer_matcher) to be used
        # without installing all optional dependencies (such as openreview).
        pass


for _name in ["dataset", "models", "preprocess", "setup", "test", "train", "utils"]:
    _optional_import(_name)
