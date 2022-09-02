from tkinter.messagebox import NO
from numpy import size
import torch
import torch.nn.functional as F
from torch import nn

from networks.layers.basic import DropPath, GroupNorm1D, GNActDWConv2d, seq_to_2d
from networks.layers.attention import MultiheadAttention, MultiheadLocalAttentionV2, MultiheadLocalAttentionV3
from networks.layers.basic import ConvGN,ResGN

def _get_norm(indim, type='ln', groups=8):
    if type == 'gn':
        return GroupNorm1D(indim, groups)
    else:
        return nn.LayerNorm(indim)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(
        F"activation should be relu/gele/glu, not {activation}.")


class LongShortTermTransformer(nn.Module):
    def __init__(self,
                 num_layers=2,
                 d_model=256,
                 self_nhead=8,
                 att_nhead=8,
                 dim_feedforward=1024,
                 emb_dropout=0.,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 droppath_scaling=False,
                 activation="gelu",
                 return_intermediate=False,
                 intermediate_norm=True,
                 final_norm=True,
                 block_version="v1"):

        super().__init__()
        self.intermediate_norm = intermediate_norm
        self.final_norm = final_norm
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.emb_dropout = nn.Dropout(emb_dropout, True)

        if block_version == "v1":
            block = LongShortTermTransformerBlock
        else:
            raise NotImplementedError

        layers = []
        for idx in range(num_layers):
            if droppath_scaling:
                if num_layers == 1:
                    droppath_rate = 0
                else:
                    droppath_rate = droppath * idx / (num_layers - 1)
            else:
                droppath_rate = droppath
            layers.append(
                block(d_model, self_nhead, att_nhead, dim_feedforward,
                      droppath_rate, lt_dropout, st_dropout, droppath_lst,
                      activation))
        self.layers = nn.ModuleList(layers)

        num_norms = num_layers - 1 if intermediate_norm else 0
        if final_norm:
            num_norms += 1
        self.decoder_norms = [
            _get_norm(d_model, type='ln') for _ in range(num_norms)
        ] if num_norms > 0 else None

        if self.decoder_norms is not None:
            self.decoder_norms = nn.ModuleList(self.decoder_norms)

    def forward(self,
                tgt,
                long_term_memories,
                short_term_memories,
                curr_id_emb=None,
                self_pos=None,
                size_2d=None):

        output = self.emb_dropout(tgt)

        intermediate = []
        intermediate_memories = []

        for idx, layer in enumerate(self.layers):
            output, memories = layer(output,
                                     long_term_memories[idx] if
                                     long_term_memories is not None else None,
                                     short_term_memories[idx] if
                                     short_term_memories is not None else None,
                                     curr_id_emb=curr_id_emb,
                                     self_pos=self_pos,
                                     size_2d=size_2d)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_memories.append(memories)

        if self.decoder_norms is not None:
            if self.final_norm:
                output = self.decoder_norms[-1](output)

            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

                if self.intermediate_norm:
                    for idx in range(len(intermediate) - 1):
                        intermediate[idx] = self.decoder_norms[idx](
                            intermediate[idx])

        if self.return_intermediate:
            return intermediate, intermediate_memories

        return output, memories


class LongShortTermTransformerBlock(nn.Module):
    def __init__(self,
                 d_model,
                 self_nhead,
                 att_nhead,
                 dim_feedforward=1024,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 activation="gelu",
                 local_dilation=1,
                 enable_corr=True):
        super().__init__()

        # Long Short-Term Attention
        self.norm1 = _get_norm(d_model)
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)

        self.long_term_attn = MultiheadAttention(d_model,
                                                 att_nhead,
                                                 use_linear=False,
                                                 dropout=lt_dropout)

        MultiheadLocalAttention = MultiheadLocalAttentionV2 if enable_corr else MultiheadLocalAttentionV3
        self.short_term_attn = MultiheadLocalAttention(d_model,
                                                       att_nhead,
                                                       dilation=local_dilation,
                                                       use_linear=False,
                                                       dropout=st_dropout)
        self.lst_dropout = nn.Dropout(max(lt_dropout, st_dropout), True)
        self.droppath_lst = droppath_lst

        # Self-attention
        self.norm2 = _get_norm(d_model)
        self.self_attn = MultiheadAttention(d_model, self_nhead)

        # Feed-forward
        self.norm3 = _get_norm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = GNActDWConv2d(dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.droppath = DropPath(droppath, batch_dim=1)
        self._init_weight()

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                long_term_memory=None,
                short_term_memory=None,
                curr_id_emb=None,
                self_pos=None,
                size_2d=(30, 30)):

        # Self-attention
        _tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(_tgt, self_pos)
        v = _tgt
        tgt2 = self.self_attn(q, k, v)[0]

        tgt = tgt + self.droppath(tgt2)

        # Long Short-Term Attention
        _tgt = self.norm2(tgt)

        curr_Q = self.linear_Q(_tgt)
        curr_K = curr_Q
        curr_V = _tgt

        local_Q = seq_to_2d(curr_Q, size_2d)

        if curr_id_emb is not None:
            global_K, global_V = self.fuse_key_value_id(
                curr_K, curr_V, curr_id_emb)
            local_K = seq_to_2d(global_K, size_2d)
            local_V = seq_to_2d(global_V, size_2d)
        else:
            global_K, global_V = long_term_memory
            local_K, local_V = short_term_memory

        tgt2 = self.long_term_attn(curr_Q, global_K, global_V)[0]
        tgt3 = self.short_term_attn(local_Q, local_K, local_V)[0]

        if self.droppath_lst:
            tgt = tgt + self.droppath(tgt2 + tgt3)
        else:
            tgt = tgt + self.lst_dropout(tgt2 + tgt3)

        # Feed-forward
        _tgt = self.norm3(tgt)

        tgt2 = self.linear2(self.activation(self.linear1(_tgt), size_2d))

        tgt = tgt + self.droppath(tgt2)

        return tgt, [[curr_K, curr_V], [global_K, global_V],
                     [local_K, local_V]]

    def fuse_key_value_id(self, key, value, id_emb):
        K = key
        V = self.linear_V(value + id_emb)
        return K, V

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class MSLongShortTermTransformer(nn.Module):
    def __init__(self,
                 num_layers=[3,1,1,1],
                 d_encoder=[24, 32, 96, 1280],
                 d_model=[256,128,64,32],
                 self_nheads=[8,4,1,1],
                 att_nheads=[8,4,1,1],
                 dims_feedforward=[1024,512,256,128],
                 global_dilations=[1,1,2,4],
                 local_dilations=[1,1,1,1],
                 memory_dilation=False,
                 emb_dropouts=[0.,0.,0.,0.],
                 droppath=[0.1,0.1,0.1,0.1],
                 lt_dropout=[0.,0.,0.,0.],
                 st_dropout=[0.,0.,0.,0.],
                 droppath_lst=[False,False,False,False],
                 droppath_scaling=[False,False,False,False],
                 activation="gelu",
                 return_intermediate=False,
                 intermediate_norm=True,
                 final_norm=True,
                 align_corners=True,
                 decode_intermediate_input=False,
                 decoder_res=False,
                 decoder_res_in=False):
        super().__init__()
        super().__init__()
        self.intermediate_norm = intermediate_norm
        self.final_norm = final_norm
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.align_corners = align_corners
        self.decode_intermediate_input = decode_intermediate_input
        
        self.emb_dropouts = nn.ModuleList()
        for i in range(len(d_model)):
            self.emb_dropouts.append(nn.Dropout(emb_dropouts[i], True))
        
        # LSTT layers
        block = MSLongShortTermTransformerBlock

        self.layers_list = nn.ModuleList()
        for s,num in enumerate(num_layers):
            layers = nn.ModuleList()
            for idx in range(num):
                if droppath_scaling[s]:
                    if num == 0 or num == 1:
                        droppath_rate = 0
                    else:
                        droppath_rate = droppath[s] * idx / (num - 1)
                else:
                    droppath_rate = droppath[s]
                layers.append(
                    block(d_model[s], self_nheads[s], att_nheads[s], dims_feedforward[s],
                        droppath_rate, lt_dropout[s], st_dropout[s], droppath_lst[s],
                        activation,global_dilation=global_dilations[s],
                        local_dilation=local_dilations[s],memory_dilation=memory_dilation))
            self.layers_list.append(layers)
        
        # decoder layers
        self.decoder_norms = nn.ModuleList()
        self.convs_in = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.convs_out = nn.ModuleList()
        
        for s,num in enumerate(num_layers):
            for i in range(num):
                if self.intermediate_norm:
                    self.decoder_norms.append(_get_norm(d_model[s], type='ln'))
                else:
                    self.decoder_norms.append(nn.Identity())
            if num >0 and not final_norm:
                self.decoder_norms[-1] = nn.Identity()
        
        for s in range(len(num_layers)-1):
            if self.decode_intermediate_input:
                if decoder_res_in:
                    self.convs_in.append(ResGN(d_model[s]*(num_layers[s]+1), d_model[s+1]))
                else:
                    self.convs_in.append(ConvGN(d_model[s]*(num_layers[s]+1), d_model[s+1],1))
            else:
                if decoder_res_in:
                    self.convs_in.append(ResGN(d_model[s+1], d_model[s+1]))
                else:
                    self.convs_in.append(ConvGN(d_model[s], d_model[s+1], 1))
            if decoder_res:
                self.convs_out.append(ResGN(d_model[s+1], d_model[s+1]))
            else:
                self.convs_out.append(ConvGN(d_model[s+1], d_model[s+1], 3))

    def forward(self,
                embs,
                long_term_memories,
                short_term_memories,
                curr_id_embs=None,
                self_pos=None,
                sizes_2d=None):
        
        
        embs = list(reversed(embs))
        output = embs[0]
        bs, c, h, w = output.size()
        output = output.view(bs, c, h * w).permute(2, 0, 1) # (B,C,H,W) -> (HW,B,C)
        all_outputs = []
        all_memories = []
        tmp_outputs = [output]
        tmp_memories = []
        
        idx = 0
        s=0
        for layer in self.layers_list[s]:
            output, memories = layer(output,
                                        long_term_memories[idx] if
                                            long_term_memories is not None else None,
                                        short_term_memories[idx] if
                                            short_term_memories is not None else None,
                                        curr_id_emb=curr_id_embs[s] if
                                            curr_id_embs is not None else None,
                                        self_pos=self_pos[s] if 
                                            self_pos is not None else None,
                                        size_2d=sizes_2d[s])
            # decoder norm
            if self.decoder_norms is not None:
                output = self.decoder_norms[idx](output)
            
            tmp_outputs.append(output)
            tmp_memories.append(memories)
            idx += 1
        if self.return_intermediate:
            all_outputs = all_outputs + tmp_outputs
            all_memories = all_memories + tmp_memories
        
        for layers in self.layers_list[1:]: # loop in scale
            if s==0 and len(tmp_outputs) == 1: # skip first scale if layer is 0
                x = embs[s+1]
            else:
                # merge lstt layer outputs
                if self.decode_intermediate_input:
                    for i in range(len(tmp_outputs)):
                        tmp_outputs[i] = tmp_outputs[i].view(sizes_2d[s][0],sizes_2d[s][1],bs,-1).permute(2,3,0,1) #(HW,B,C) -> (B,C,H,W)
                    x = torch.cat(tmp_outputs, dim=1)
                else:
                    x = tmp_outputs[-1].view(sizes_2d[s][0],sizes_2d[s][1],bs,-1).permute(2,3,0,1)
                # down channel
                x = F.relu(self.convs_in[s](x)) 
                # upscale
                x = F.interpolate(x,
                            size=sizes_2d[s+1],
                            mode="bilinear",
                            align_corners=self.align_corners)
                # add next scale feature
                x = F.relu(self.convs_out[s](embs[s+1] + x))
            
            # input to next scale
            s += 1
            output = x

            bs, c, h, w = output.size()
            output = output.view(bs, c, h * w).permute(2, 0, 1) # (B,C,H,W) -> (HW,B,C)
            tmp_outputs = [output]
            tmp_memories = []

            for layer in layers: # loop in LSTT layer
                output, memories = layer(output,
                                        long_term_memories[idx] if
                                            long_term_memories is not None else None,
                                        short_term_memories[idx] if
                                            short_term_memories is not None else None,
                                        curr_id_emb=curr_id_embs[s] if
                                            curr_id_embs is not None else None,
                                        self_pos=self_pos[s] if
                                            self_pos is not None else None,
                                        size_2d=sizes_2d[s])
                # decoder norm
                if self.decoder_norms is not None:
                    output = self.decoder_norms[idx](output)
                
                tmp_outputs.append(output)
                tmp_memories.append(memories)
                idx += 1

            if self.return_intermediate:
                all_outputs = all_outputs + tmp_outputs
                all_memories = all_memories + tmp_memories
                    
                
        
        if self.decode_intermediate_input:
            for i in range(len(tmp_outputs)):
                tmp_outputs[i] = tmp_outputs[i].view(sizes_2d[s][0],sizes_2d[s][1],bs,-1).permute(2,3,0,1) #(B,C,H,W)
            output = torch.cat(tmp_outputs, dim=1)
        else:
            output = tmp_outputs[-1].view(sizes_2d[s][0],sizes_2d[s][1],bs,-1).permute(2,3,0,1)
        all_outputs.append(output)

        if self.return_intermediate:
            return all_outputs, all_memories

        return output, memories


class MSLongShortTermTransformerBlock(nn.Module):
    def __init__(self,
                 d_model,
                 self_nhead,
                 att_nhead,
                 dim_feedforward=1024,
                 droppath=0.1,
                 lt_dropout=0.,
                 st_dropout=0.,
                 droppath_lst=False,
                 activation="gelu",
                 global_dilation=1,
                 local_dilation=1,
                 memory_dilation=False,
                 enable_corr=True):
        super().__init__()

        self.d_model = d_model
        self.att_nhead = att_nhead
        self.memory_dilation=memory_dilation

        # Self-attention
        self.norm1 = _get_norm(d_model)
        self.self_attn = MultiheadAttention(d_model, self_nhead)

        # Long Short-Term Attention
        self.global_dilation = global_dilation
        self.norm2 = _get_norm(d_model)
        self.linear_QV = nn.Linear(d_model, 2 * d_model)
        self.linear_ID_KV = nn.Linear(d_model, d_model + att_nhead)

        self.long_term_attn = MultiheadAttention(d_model,
                                                 att_nhead,
                                                 use_linear=False,
                                                 dropout=lt_dropout)

        MultiheadLocalAttention = MultiheadLocalAttentionV2 if enable_corr else MultiheadLocalAttentionV3
        self.short_term_attn = MultiheadLocalAttention(d_model,
                                                       att_nhead,
                                                       dilation=local_dilation,
                                                       use_linear=False,
                                                       dropout=st_dropout)
        self.lst_dropout = nn.Dropout(max(lt_dropout, st_dropout), True)
        self.droppath_lst = droppath_lst

        # Feed-forward
        self.norm3 = _get_norm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = GNActDWConv2d(dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.droppath = DropPath(droppath, batch_dim=1)
        self._init_weight()

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(self,
                tgt,
                long_term_memory=None,
                short_term_memory=None,
                curr_id_emb=None,
                self_pos=None,
                size_2d=(30, 30)):

        # Self-attention
        _tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(_tgt, self_pos)
        v = _tgt
        if self.global_dilation > 1:
            k = k[::self.global_dilation,:,:]
            v = v[::self.global_dilation,:,:]
        tgt2 = self.self_attn(q, k, v)[0]

        tgt = tgt + self.droppath(tgt2)

        # Long Short-Term Attention
        _tgt = self.norm2(tgt)

        curr_QV = self.linear_QV(_tgt)
        curr_QV = torch.split(curr_QV, self.d_model, dim=2)
        curr_Q = curr_K = curr_QV[0]
        curr_V = curr_QV[1]

        local_Q = seq_to_2d(curr_Q, size_2d)

        if curr_id_emb is not None:
            global_K, global_V = self.fuse_key_value_id(
                curr_K, curr_V, curr_id_emb)
            local_K = seq_to_2d(global_K, size_2d)
            local_V = seq_to_2d(global_V, size_2d)

            if self.global_dilation>1 and self.memory_dilation:
                nhw,bs,c = global_K.shape
                n = nhw // (size_2d[0] * size_2d[1])
                d = self.global_dilation
                unfold_K = global_K.view(n,size_2d[0],size_2d[1],bs,c)
                unfold_V = global_V.view(n,size_2d[0],size_2d[1],bs,c)
                global_K = unfold_K[:,::d,::d,:,:].reshape(-1,bs,c)
                global_V = unfold_V[:,::d,::d,:,:].reshape(-1,bs,c)
        else:
            global_K, global_V = long_term_memory
            local_K, local_V = short_term_memory
        
        
        if self.memory_dilation:
            tgt2 = self.long_term_attn(curr_Q, global_K, global_V)[0]
        else:
            if self.global_dilation>1:
                nhw,bs,c = global_K.shape
                n = nhw // (size_2d[0] * size_2d[1])
                d = self.global_dilation
                unfold_K = global_K.view(n,size_2d[0],size_2d[1],bs,c)
                unfold_V = global_V.view(n,size_2d[0],size_2d[1],bs,c)
                dilated_K = unfold_K[:,::d,::d,:,:].reshape(-1,bs,c)
                dilated_V = unfold_V[:,::d,::d,:,:].reshape(-1,bs,c)
            else:
                dilated_K,dilated_V = global_K,global_V
            tgt2 = self.long_term_attn(curr_Q, dilated_K, dilated_V)[0]

        tgt3 = self.short_term_attn(local_Q, local_K, local_V)[0]

        if self.droppath_lst:
            tgt = tgt + self.droppath(tgt2 + tgt3)
        else:
            tgt = tgt + self.lst_dropout(tgt2 + tgt3)

        # Feed-forward
        _tgt = self.norm3(tgt)

        tgt2 = self.linear2(self.activation(self.linear1(_tgt), size_2d))

        tgt = tgt + self.droppath(tgt2)

        return tgt, [[curr_K, curr_V], [global_K, global_V],
                     [local_K, local_V]]

    def fuse_key_value_id(self, key, value, id_emb):
        ID_KV = self.linear_ID_KV(id_emb)
        ID_K, ID_V = torch.split(ID_KV, [self.att_nhead, self.d_model], dim=2)
        bs = key.size(1)
        K = key.view(-1, bs, self.att_nhead, self.d_model //
                     self.att_nhead) * (1 + torch.tanh(ID_K)).unsqueeze(-1)
        K = K.view(-1, bs, self.d_model)
        V = value + ID_V
        return K, V

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


