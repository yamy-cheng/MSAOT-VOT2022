class DefaultModelConfig():
    def __init__(self):
        self.MODEL_NAME = 'AOTDefault'

        self.MODEL_VOS = 'aot'
        self.MODEL_ENGINE = 'aotengine'
        self.MODEL_ALIGN_CORNERS = True
        self.MODEL_ENCODER = 'mobilenetv2'
        self.MODEL_ENCODER_PRETRAIN = './pretrain_models/mobilenet_v2-b0353104.pth'
        self.MODEL_ENCODER_DIM = [24, 32, 96, 1280]  # 4x, 8x, 16x, 16x
        self.MODEL_ENCODER_EMBEDDING_DIM = 256
        self.MODEL_DECODER_INTERMEDIATE_LSTT = True
        self.MODEL_FREEZE_BN = True
        self.MODEL_FREEZE_BACKBONE = False
        self.MODEL_MAX_OBJ_NUM = 10
        self.MODEL_SELF_HEADS = 8
        self.MODEL_ATT_HEADS = 8
        self.MODEL_LSTT_NUM = 1
        self.MODEL_EPSILON = 1e-5
        self.MODEL_USE_PREV_PROB = False

        self.TRAIN_LONG_TERM_MEM_GAP = 9999

        self.TEST_LONG_TERM_MEM_GAP = 9999

        # multi-scale param
        self.MODEL_MS_LSTT_NUMS = [2,1,1,1]
        self.MODEL_MS_ENCODER_EMBEDDING_DIMS = [256,256,128,128]
        self.MODEL_MS_SCALES = [16,16,8,4]
        self.MODEL_MS_SELF_HEADS = [8,1,1,1]
        self.MODEL_MS_ATT_HEADS = [8,1,1,1]
        self.MODEL_MS_FEEDFOWARD_DIMS = [1024,1024,512,512]
        self.MODEL_MS_GLOBAL_DILATIONS = [1,1,2,4]
        self.MODEL_MS_LOCAL_DILATIONS = [1,1,1,1]
        self.TRAIN_MS_LSTT_EMB_DROPOUTS = [0.,0.,0.,0.]
        self.MODEL_MS_SHARE_ID = False
        self.MODEL_MS_SHARE_ID_SCALE = 0
        self.MODEL_DECODER_RES = False
        self.MODEL_DECODER_RES_IN = False

        self.TRAIN_MS_LSTT_DROPPATH = [0.1,0.1,0.1,0.1]
        self.TRAIN_MS_LSTT_DROPPATH_SCALING = [False,False,False,False]
        self.TRAIN_MS_LSTT_DROPPATH_LST = [False,False,False,False]
        self.TRAIN_MS_LSTT_LT_DROPOUT = [0.,0.,0.,0.]
        self.TRAIN_MS_LSTT_ST_DROPOUT = [0.,0.,0.,0.]
        self.TRAIN_MS_LSTT_MEMORY_DILATION = False
        
        
