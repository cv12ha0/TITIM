# Attacks
from .attack.fake_trigger import *
from .attack.square import *
from .attack.patch import *
from .attack.badnets import *
from .attack.blended import *
from .attack.styled import *
from .attack.wanet import *
from .attack.sig import *
from .attack.compress import *


# Defenses
from .defense.activation_clustering import *
from .defense.spectral_signature import * 
from .defense.strip import *
from .defense.scale_up import *
from .defense.neural_cleanse import *
from .defense.k_arm import *
from .defense.feature_re import *
from .defense.abl import *


def dump_trigger(trigger, path):
    os.makedirs(os.path.join(path), exist_ok=True)
    fname = os.path.join(path, 'trigger.pkl')
    pickle.dump(trigger, open(fname, 'wb'))
    try:
        trigger.visualize(os.path.join(path, 'trigger_imgs'))
    except AttributeError:
        pass
    print("Trigger saved to {}".format(fname))


def dump_trigger_config(trigger, path):
    import json
    os.makedirs(os.path.join(path), exist_ok=True)
    fname = os.path.join(path, 'trigger_config.json')
    with open(fname, 'w') as f:
        json.dump(trigger.config, f, indent=4)
    print("Trigger configuration saved to {}".format(fname))


if __name__ == '__main__':
    pass
