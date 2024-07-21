from nbtr.tokenizer.train import Trainer as TokTrainer
from nbtr.train.trainer import TrainerConfig
import argparse

def main(ds_repo_id, repo_id, out_dir):
    tok_trainer = TokTrainer(ds_repo_id, repo_id, out_dir)
    tok_trainer.train()
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_repo_id", type=str, required=True, help="Dataset repository id")
    parser.add_argument("--repo_id", type=str, required=True, help="Model repository id")
    parser.add_argument("--out_dir", type=str, required=False, default=None, help="Leave empty to infer from repo_id")
    args = parser.parse_args()

    ds_repo_id = args.ds_repo_id
    assert ds_repo_id is not None
    repo_id = args.repo_id
    assert repo_id is not None
    out_dir = repo_id.split("/")[-1]
    assert out_dir is not None
    
    main(ds_repo_id, repo_id, out_dir)