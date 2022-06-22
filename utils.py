import argparse
import yaml
import os

def get_config_from_yaml(yaml_file_paths):
    with open(yaml_file_paths, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

def save_config(config):
    filename = config["result_dir"] + \
        "training_config_lwin_{}_embed_dim_{}_lr_{}_.json".format(
            config["l_win"], config["embed_dim"], config["lr"])
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)

