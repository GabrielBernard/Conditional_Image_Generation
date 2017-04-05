import sys
import argparse

try:
    from models.utils import data_utils
except ImportError:
    from utils import data_utils

def main():
    parser = argparse.ArgumentParser()
    requiredArg = parser.add_argument_group('Required Arguments')
    requiredArg.add_argument("-dp", "--datapath", help="Path to the data", type=str, required=True)
    parser.add_argument("-sp", "--savepath", help="Path to folder to save calculations", type=str)
    args = parser.parse_args()
    print(args.datapath)
    print(args.savepath)
    data_utils.crop_data(args.datapath, args.savepath)

if __name__ == '__main__':
    main()
