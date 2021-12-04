from facenet_pytorch import InceptionResnetV1
from .iresnet import iresnet18, iresnet34, iresnet50, \
                     iresnet100, iresnet200
from .iresnet2060 import iresnet2060

def inception_resnet_v1(*args, **kwargs):
    return InceptionResnetV1(*args, **kwargs)

NAME_MODEL_MAP = {
    'resnet18'      :   iresnet18,
    'resnet34'      :   iresnet34,
    'resnet50'      :   iresnet50,
    'resnet100'     :   iresnet100,
    'resnet200'     :   iresnet200,
    'resnet2060'    :   iresnet2060,
    'facenet'       :   inception_resnet_v1     
}

def get_model(name, *args, **kwargs):
    if name in NAME_MODEL_MAP:
        return NAME_MODEL_MAP[name](*args, **kwargs)
    else:
        raise NotImplementedError