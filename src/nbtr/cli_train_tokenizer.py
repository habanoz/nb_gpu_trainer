from nbtr.tokenizer.train import Trainer as TokTrainer
from nbtr.train.trainer import TrainerConfig
import argparse

def main(ds_repo_id, repo_id, out_dir, vocab_exp, ratio) :
    tok_trainer = TokTrainer(ds_repo_id, repo_id, out_dir, vocab_exp, ratio)
    tok_trainer.train()
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_repo_id", type=str, required=True, help="Dataset repository id")
    parser.add_argument("--repo_id", type=str, required=True, help="Model repository id")
    parser.add_argument("--out_dir", type=str, required=False, default=None, help="Leave empty to infer from repo_id")
    parser.add_argument("--vocab_exp", type=int, required=False, default=13, help="Vocabulary size as a power of 2.")
    parser.add_argument("--ratio", type=float, required=False, default=1.0, help="If you do not want to use full dataset to train the tokenizer, specify what ratio of dataset to use. e.g. 0.5 to use half.")
    args = parser.parse_args()

    ds_repo_id = args.ds_repo_id
    assert ds_repo_id is not None
    repo_id = args.repo_id
    assert repo_id is not None
    out_dir = repo_id.split("/")[-1]
    assert out_dir is not None
    vocab_exp = args.vocab_exp
    ratio = args.ratio
    assert 0<ratio<=1
    
    main(ds_repo_id, repo_id, out_dir, vocab_exp, ratio)