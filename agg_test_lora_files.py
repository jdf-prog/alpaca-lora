import json
import os
import fire
import regex as re
from pathlib import Path

def main(
    output_file:str
):
    output_file = Path(output_file)
    agg_data = []
    for file in output_file.parent.iterdir():
        shard_file = None
        if re.match(rf"{output_file.name}.shard-[0-9]+$", file.name):
            shard_file = file
        if shard_file is None:
            continue
        print(f"Aggregating {shard_file}")
        with open(shard_file, 'r') as f:
            data = json.load(f)
        agg_data.extend(data)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    fire.Fire(main)