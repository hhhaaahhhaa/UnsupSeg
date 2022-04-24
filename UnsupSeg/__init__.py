import os
import shutil
from . import Define

from .SegmentModel import load_model_from_tag
from .Tags import ModelTag


__version__ = "v0.0.0"


repo_path = os.path.dirname(os.path.dirname(__file__))

Define.REPO_PATH = repo_path
