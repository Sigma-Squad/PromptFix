from datasets import load_dataset, Dataset
import argparse

parser = argparse.ArgumentParser(description="Dataset file name")
parser.add_argument ('--start', type=int, help='Start index')
parser.add_argument ('--end', type=int, help='End index')
args = parser.parse_args()

start, end = args.start, args.end
assert start <= end, "start must be less than or equal to end"

# load the dataset with data streaming
ds = load_dataset("yeates/PromptFixData", split="train", cache_dir="./hf_cache", streaming=True)  

# collect the required number of samples
samples = []
for i, sample in enumerate(ds):
    if i >= start and i < end:
        samples.append(sample)

    if i + 1 >= end:
        break

collected_ds = Dataset.from_list(samples)

# save the collected samples locally
collected_ds.save_to_disk(f"dataset/{start}_{end}.hf")
