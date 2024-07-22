from nbtr.tokenizer.tokenizer import Tokenizer
from nbtr.train.trainer import TrainerConfig
import argparse

def main(repo_id, ds_repo_id, data_dir):
    tokenizer = Tokenizer.from_pretrained(repo_id)
    tokenizer.encode_ds_from_hub(ds_repo_id, data_dir)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="Model repository id")
    parser.add_argument("--ds_repo_id", type=str, required=True, help="Dataset repository id")
    parser.add_argument("--data_dir", type=str, required=False, default=None, help="Leave empty to infer from ds_repo_id")
    args = parser.parse_args()

    ds_repo_id = args.ds_repo_id
    assert ds_repo_id is not None
    repo_id = args.repo_id
    assert repo_id is not None
    
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = ds_repo_id.split("/")[-1]

    main(repo_id, ds_repo_id, data_dir)