if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    import argparse
    import os
    import dotenv

    from src.signal_editor.app import main

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dev", action="store_true")
    parser.add_argument("-p", "--profile", action="store_true")
    parser.add_argument("-c", "--console", action="store_true")
    parser.add_argument("-a", "--antialias", action="store_true")
    args = parser.parse_args()

    os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

    if args.dev:
        os.environ["QT_LOGGING_RULES"] = "qt.pyside.libpyside.warning=true"
        os.environ["DEV_MODE"] = "1"
    if args.profile:
        os.environ["PROFILE"] = "1"
    if args.console:
        os.environ["ENABLE_CONSOLE"] = "1"
    if args.antialias:
        os.environ["PG_ANTIALIAS"] = "1"

    dotenv.load_dotenv(".env")

    main()
