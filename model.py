from dataset import *
from Unet import *
import Unet
def create_model(params):

    # model = getattr(ternausnet.models, params["model"])(pretrained=True)
    model = getattr(Unet, params["model"])(pretrained=True)
    model = model.to(params["device"])
    return model