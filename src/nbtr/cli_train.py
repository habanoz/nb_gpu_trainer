from nbtr.train.trainer import TrainerConfig
from nbtr.train.hf_backed_trainer import HFBackedTrainer
from nbtr.model.hf_model import HfModel
from nbtr.model.hf_trainer_config import HfTrainerConfig
from nbtr.model.hf_model_config import HfModelConfig
from nbtr.model.gpt2 import GPTConfig
import argparse


def main_with_repo_id(repo_id):
    # prepare model
    hf_model = HfModel.from_pretrained(repo_id)
    hf_model.to('cuda')

    # prepare trainer
    hf_trained_config = HfTrainerConfig.from_pretrained(repo_id=repo_id)
    trainer = HFBackedTrainer(hf_trainer_config=hf_trained_config)

    # train
    trainer.train(hf_model=hf_model)

    print("Done. Repo id:", repo_id)


def main_with_config(trainer_config_file, model_config_file):
    trainer_config = TrainerConfig.from_yaml(trainer_config_file)

    # prepare model
    gpt_config = GPTConfig.from_yaml(model_config_file)
    hf_model_config = HfModelConfig(gpt_config=gpt_config)
    hf_model = HfModel(config=hf_model_config)
    hf_model.to('cuda')

    # prepare trainer
    hf_trainer_config = HfTrainerConfig(trainer_config=trainer_config)
    trainer = HFBackedTrainer(hf_trainer_config=hf_trainer_config)

    # train
    trainer.train(hf_model=hf_model)

    print("Done. Config file:", trainer_config_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default=None,
                        help="Model repository id")
    parser.add_argument("--trainer_config_file", "-C", type=str,
                        required=False, default=None, help="Trainer config file")
    parser.add_argument("--model_config_file", "-M", type=str,
                        required=False, default=None, help="Model config file")
    args = parser.parse_args()

    repo_id = args.repo_id
    trainer_config_file = args.trainer_config_file
    model_config_file = args.model_config_file

    if trainer_config_file is None or model_config_file is None:
        if repo_id is None:
            raise ValueError(
                "Repo id must be provided if configuration files are not provided!")

    if repo_id is not None:
        main_with_repo_id(repo_id, trainer_config_file, model_config_file)
    else:
        main_with_config(trainer_config_file, model_config_file)
