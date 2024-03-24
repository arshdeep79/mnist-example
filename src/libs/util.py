import yaml
def loadYamlFile(pathToFile):
    with open(pathToFile, 'r') as file:
        return yaml.safe_load(file)
    