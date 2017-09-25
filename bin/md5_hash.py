#!/usr/bin/env python3
import hashlib
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    print(hashlib.md5(open(args.filename, 'rb').read()).hexdigest())
