from argparse import ArgumentParser
import matplotlib.pyplot as plt
from os import walk, listdir, path
import os


def distribution(path):
#    w = walk(path)
#    for (dirpath, dirnames, filenames) in w:
#        print(dirpath, dirnames, filenames)
    for path_file in listdir(path):
        # check if current path is a file
        print(os.path.join(path, path_file))
    return


def main():
    parser = ArgumentParser()
    parser.add_argument("path",
                        type=str,
                        help="Path of the folder to analyze")
    args = parser.parse_args()
    try:
        args = vars(args)
        distribution(**args)
    except Exception as e:
        print(str(e))
        parser.print_help()


if __name__ == "__main__":
    main()
