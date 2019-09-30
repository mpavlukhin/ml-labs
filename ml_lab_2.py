import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='ML Lab 2'
    )

    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()


if __name__ == "__main__":
    main()
