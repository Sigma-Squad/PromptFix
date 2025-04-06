from datasets import load_from_disk
import argparse

parser = argparse.ArgumentParser(description="Dataset file name")
parser.add_argument ('-f', type=str, help='The local dataset to load')
args = parser.parse_args()

ds = load_from_disk(args.f)
print(ds[1])