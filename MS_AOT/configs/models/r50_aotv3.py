from .default import DefaultModelConfig


class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'R50_AOTv3'
        self.MODEL_VOS = 'aotv3'
        self.MODEL_ENGINE = 'aotv3engine'
        self.MODEL_ENCODER = 'resnet50'
        self.MODEL_ENCODER_DIM = [256, 512, 1024, 1024]  # 4x, 8x, 16x, 16x
        
        self.MODEL_DECODER_INTERMEDIATE_LSTT = True

        self.TRAIN_LONG_TERM_MEM_GAP = 2

        self.TEST_LONG_TERM_MEM_GAP = 10
