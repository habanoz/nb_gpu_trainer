import yaml
from dataclasses import dataclass

def from_yaml(csl:type[dataclass], config_file:str):
    with open(config_file) as f:
        doc = yaml.safe_load(f)
    
    return csl(**doc)