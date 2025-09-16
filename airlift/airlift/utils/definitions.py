import os
from pathlib import Path

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../..'))
ROOT_PATH = Path(ROOT_DIR)

TEST_MODE = False