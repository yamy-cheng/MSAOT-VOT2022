import imp
import torch.nn as nn
import torch.nn.functional as F

from networks.encoders import build_encoder
from networks.decoders import build_decoder
from networks.layers.position import PositionEmbeddingSine
from networks.layers.transformer import MSLongShortTermTransformer
from networks.layers.basic import ConvGN


class AOTv3(nn.Module):
    def __init__(self, cfg, encoder='mobilenetv2', decoder='fpn'):
        super().__init__()

        self.cfg = cfg
        self.max_obj_num = cfg.MODEL_MAX_OBJ_NUM
        self.epsilon = cfg.MODEL_EPSILON

        self.encoder = build_encoder(encoder,
                                     frozen_bn=cfg.MODEL_FREEZE_BN,
                                     freeze_at=cfg.TRAIN_ENCODER_FREEZE_AT)
        
        self.adapters = nn.ModuleList()
        for s in range(len(cfg.MODEL_ENCODER_DIM)):
            self.adapters.append(nn.Conv2d(cfg.MODEL_ENCODER_DIM[-(s+1)], cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[s], 1))

        self.pos_generators = nn.ModuleList()
        for s in range(len(cfg.MODEL_ENCODER_DIM)):
            self.pos_generators.append(PositionEmbeddingSine(
                cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[s]//2, normalize=True))

        self.MSLSTT = MSLongShortTermTransformer(
            cfg.MODEL_MS_LSTT_NUMS,
            cfg.MODEL_ENCODER_DIM,
            cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS,
            cfg.MODEL_MS_SELF_HEADS,
            cfg.MODEL_MS_ATT_HEADS,
            dims_feedforward=cfg.MODEL_MS_FEEDFOWARD_DIMS,
            global_dilations=cfg.MODEL_MS_GLOBAL_DILATIONS,
            local_dilations=cfg.MODEL_MS_LOCAL_DILATIONS,
            memory_dilation=cfg.TRAIN_MS_LSTT_MEMORY_DILATION,
            emb_dropouts=cfg.TRAIN_MS_LSTT_EMB_DROPOUTS,
            droppath=cfg.TRAIN_MS_LSTT_DROPPATH,
            lt_dropout=cfg.TRAIN_MS_LSTT_LT_DROPOUT,
            st_dropout=cfg.TRAIN_MS_LSTT_ST_DROPOUT,
            droppath_lst=cfg.TRAIN_MS_LSTT_DROPPATH_LST,
            droppath_scaling=cfg.TRAIN_MS_LSTT_DROPPATH_SCALING,
            intermediate_norm=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            return_intermediate=True,
            align_corners=cfg.MODEL_ALIGN_CORNERS,
            decode_intermediate_input=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
            decoder_res=cfg.MODEL_DECODER_RES,
            decoder_res_in=cfg.MODEL_DECODER_RES_IN)
        
        
        self.patch_wise_id_banks = nn.ModuleList()
        self.id_norms = nn.ModuleList()
        scales = cfg.MODEL_MS_SCALES
        for i,s in enumerate(scales):
            if cfg.MODEL_ALIGN_CORNERS:
                self.patch_wise_id_banks.append(nn.Conv2d(
                    cfg.MODEL_MAX_OBJ_NUM + 1,
                    cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[i],
                    kernel_size=s+1,
                    stride=s,
                    padding=s//2))
            else:
                self.patch_wise_id_banks.append(nn.Conv2d(
                    cfg.MODEL_MAX_OBJ_NUM + 1,
                    cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[i],
                    kernel_size=s,
                    stride=s,
                    padding=0))
            self.id_norms.append(nn.LayerNorm(cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[i]))
        self.id_dropout = nn.Dropout(cfg.TRAIN_LSTT_ID_DROPOUT, True)
        
        if cfg.MODEL_DECODER_INTERMEDIATE_LSTT:
            self.conv_output1 = ConvGN(cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[-1] * (cfg.MODEL_MS_LSTT_NUMS[-1] + 1),
                                    cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[-1], 3)
        else:
            self.conv_output1 = ConvGN(cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[-1],
                                    cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[-1], 3)
        self.conv_output2 = nn.Conv2d(cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[-1], cfg.MODEL_MAX_OBJ_NUM + 1, 1)
        
        if cfg.TRAIN_INTERMEDIATE_PRED_LOSS:
            decoder_indim = cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[0] * \
                (cfg.MODEL_MS_LSTT_NUMS[0] + 1) \
                if cfg.MODEL_DECODER_INTERMEDIATE_LSTT else cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[0]
            self.decoder = build_decoder(
                decoder,
                in_dim=decoder_indim,
                out_dim=cfg.MODEL_MAX_OBJ_NUM + 1,
                decode_intermediate_input=cfg.MODEL_DECODER_INTERMEDIATE_LSTT,
                hidden_dim=cfg.MODEL_ENCODER_EMBEDDING_DIM,
                shortcut_dims=cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS,
                align_corners=cfg.MODEL_ALIGN_CORNERS,
                use_adapters=False)
        
        
        self._init_weight()

    def get_pos_embs(self, xs):
        pos_embs = []
        for i,generator in enumerate(self.pos_generators):
            pos_emb = generator(xs[-(i+1)])
            pos_embs.append(pos_emb)
        return pos_embs

    def get_id_embs(self,x):
        id_embs = []
        for id_bank,id_norm in zip(self.patch_wise_id_banks,self.id_norms):
            id_emb = id_bank(x)
            id_emb = id_norm(id_emb.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
            id_emb = self.id_dropout(id_emb)
            id_embs.append(id_emb)
        if self.cfg.MODEL_MS_SHARE_ID:
            s = self.cfg.MODEL_MS_SHARE_ID_SCALE
            for i in range(0,len(id_embs)):
                if i != s:
                    id_embs[i] = F.interpolate(id_embs[s],id_embs[i].shape[2:],
                        mode='nearest')
        return id_embs

    def encode_image(self, img):
        xs = self.encoder(img)
        for i,adapter in enumerate(self.adapters):
            xs[-(i+1)] = adapter(xs[-(i+1)])
        return xs
    
    def decode_id_logits(self, lstt_embs):
        output = F.relu(self.conv_output1(lstt_embs[-1]))
        output = self.conv_output2(output)
        return output
    def decode_med_logits(self,lstt_med_embs,encoder_embs):
        n, c, h, w = encoder_embs[-1].size()
        decoder_inputs = []
        for emb in lstt_med_embs:
            decoder_inputs.append(emb.view(h, w, n, c).permute(2, 3, 0, 1))
        med_pred = self.decoder(decoder_inputs, encoder_embs)
        return med_pred
    
    def LSTT_forward(self,
                     curr_embs,
                     long_term_memories,
                     short_term_memories,
                     curr_id_embs=None,
                     pos_embs=None,
                     sizes_2d: list=[(30,30),(30,30),(59,59),(117,117)]):
            
        lstt_embs, lstt_memories = self.MSLSTT(curr_embs, long_term_memories,
                                             short_term_memories, curr_id_embs,
                                             pos_embs, sizes_2d)
        lstt_curr_memories, lstt_long_memories, lstt_short_memories = zip(
            *lstt_memories)
        return lstt_embs, lstt_curr_memories, lstt_long_memories, lstt_short_memories
   

    def _init_weight(self):
        for adapter in self.adapters:
            nn.init.xavier_uniform_(adapter.weight)
        for s,id_bank in enumerate(self.patch_wise_id_banks):
            nn.init.orthogonal_(
                id_bank.weight.view(
                    self.cfg.MODEL_MS_ENCODER_EMBEDDING_DIMS[s], -1).permute(0, 1),
                gain=17**-2 if self.cfg.MODEL_ALIGN_CORNERS else 16**-2)
        nn.init.xavier_uniform_(self.conv_output2.weight)
