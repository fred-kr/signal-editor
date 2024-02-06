if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    import argparse
    import os

    from src.signal_editor.app import main

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dev", action="store_true")
    parser.add_argument("-c", "--console", action="store_true")
    parser.add_argument("-a", "--antialias", action="store_true")
    args = parser.parse_args()

    os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

    main(args.dev, args.antialias, args.console)
