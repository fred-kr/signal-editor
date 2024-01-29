import argparse
import os
import polars as pl

from src.signal_editor.app import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dev", action="store_true")
    parser.add_argument("-a", "--antialias", action="store_true")
    args = parser.parse_args()

    os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

    pl.Config.activate_decimals(True)

    main(args.dev, args.antialias)
