import argparse

from src.sea_signals.app import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dev", action="store_true")
    parser.add_argument("-a", "--antialias", action="store_true")
    args = parser.parse_args()

    main(args.dev, args.antialias)
