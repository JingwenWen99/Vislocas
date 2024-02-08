import argparse
import os
import sys


def parse_args():

    parser = argparse.ArgumentParser(description="Provide training and testing arguments.")

    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()
