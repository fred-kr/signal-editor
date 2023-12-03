import os
import sys

from src.sea_signals.app import main

if getattr(sys, "frozen", False):
    working_dir = os.path.dirname(sys.executable)
else:
    working_dir = os.path.dirname(os.path.abspath(__file__))

APP_WORKING_DIR = working_dir

if __name__ == "__main__":
    main(app_wd=APP_WORKING_DIR, dev_mode=False)
