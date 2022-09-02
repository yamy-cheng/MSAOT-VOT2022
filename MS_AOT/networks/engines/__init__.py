from networks.engines.aot_engine import AOTEngine, AOTInferEngine
from networks.engines.aotv3_engine import AOTv3Engine, AOTv3InferEngine


def build_engine(name, phase='train', **kwargs):

    if name == 'aotengine':
        if phase == 'train':
            return AOTEngine(**kwargs)
        elif phase == 'eval':
            return AOTInferEngine(**kwargs)
        else:
            raise NotImplementedError
    elif name == 'aotv3engine':
        if phase == 'train':
            return AOTv3Engine(**kwargs)
        elif phase == 'eval':
            return AOTv3InferEngine(**kwargs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
