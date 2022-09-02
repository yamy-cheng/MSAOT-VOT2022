from networks.models.aot import AOT
from networks.models.aotv3 import AOTv3


def build_vos_model(name, cfg, **kwargs):

    if name == 'aot':
        return AOT(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    elif name == 'aotv3':
        return AOTv3(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    else:
        raise NotImplementedError
