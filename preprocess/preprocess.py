import argparse

import yaml

from preprocessor import ljspeech, vctk, vctk_trim


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.preprocess(config)

    if 'VCTK' in config['dataset']:
        # vctk.preprocess(config)
        vctk_trim.preprocess(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/LJSpeech/preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    
    main(config)
    
    
