from nbtr.tokenizer.tokenizer import Tokenizer
import argparse

def main(from_repo_id, to_repo_id):
    Tokenizer.copy_pretrained(from_repo_id, to_repo_id)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_repo_id", type=str, required=True, help="Repo to copy tokinizer from.")
    parser.add_argument("--to_repo_id", type=str, required=True, help="New repo to copy tokenizer to.")
    args = parser.parse_args()

    from_repo_id = args.from_repo_id
    to_repo_id = args.to_repo_id

    assert "/" in from_repo_id
    assert "/" in to_repo_id
    
    main(from_repo_id, to_repo_id)