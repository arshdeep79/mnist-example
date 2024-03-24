import yaml
import libs.util as util 
import libs.model as model
import libs.checkpoints as checkpoints
from dotmap import DotMap
from PIL import Image 

# Do a training run for the model
def train(overrideConfig={}):
    # load the local config file first
    defaultConfig = util.loadYamlFile('./config.yml')
    config = DotMap({**defaultConfig, **overrideConfig})
    model.train(config)
    

# infer an image on mdoel
def infer(image):
    checkpoint = checkpoints.loadLastCheckpoint()
    if not checkpoints:
        return None
    
    return model.infer(image, checkpoint['modelState'])
