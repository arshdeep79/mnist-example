import argparse
import yaml

from libs  import modelInterface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MNIST model")
    parser.add_argument("--config", type=str, help="Config file")
    args = parser.parse_args()
    if not args.config:
        print('Requires config file "---config path/to/config.yaml"')
        exit()
    with open(args.config) as configFileContent:    
        config = yaml.safe_load(configFileContent)

    modelInterface.train(config)