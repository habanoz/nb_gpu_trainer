from nbtr.tokenizer.tokenizer import Tokenizer
from nbtr.train.trainer import TrainerConfig
import argparse

def main(repo_id, ds_repo_id, ds_repo_name, split, data_dir, prefix, limit, shard_size):
    tokenizer = Tokenizer.from_pretrained(repo_id)
    tokenizer.encode_limited_stream_ds_from_hub(ds_repo_id,ds_repo_name, split, data_dir, prefix, limit, shard_size)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="Model repository id")
    parser.add_argument("--ds_repo_id", type=str, required=True, help="Dataset repository id")
    parser.add_argument("--ds_repo_name", type=str, required=True, help="Dataset repository name")
    parser.add_argument("--split", type=str, required=True, help="Dataset repository split")
    parser.add_argument("--data_dir", type=str, required=False, default=None, help="Leave empty to infer from ds_repo_id")
    parser.add_argument("--limit", type=int, required=False, default=None, help="token limit")
    parser.add_argument("--shard_size", type=int, required=True, default=None, help="Maximum number of tokens a data file can have")
    parser.add_argument("--prefix", type=str, required=False, default="", help="Shard file names will in the following format: {prefix}_{split}_{index}.bin")
    args = parser.parse_args()

    ds_repo_id = args.ds_repo_id
    assert ds_repo_id is not None
    repo_id = args.repo_id
    assert repo_id is not None
    ds_repo_name = args.ds_repo_name
    split = args.split
    shard_size = args.shard_size
    prefix = args.prefix
    limit = args.limit
    
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = ds_repo_id.split("/")[-1]
        if ds_repo_name:
            data_dir += "_"+ds_repo_name

    main(repo_id, ds_repo_id, ds_repo_name, split, data_dir, prefix, limit, shard_size)