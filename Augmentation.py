from argparse import ArgumentParser
from Distribution import count_files_folder

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
