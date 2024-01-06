import argparse
import os

from src.signal_editor.app import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dev", action="store_true")
    parser.add_argument("-a", "--antialias", action="store_true")
    args = parser.parse_args()

    os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"  # Set environment variable

    main(args.dev, args.antialias)
