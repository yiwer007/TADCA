import math
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from modules.InfoNCE import InfoNCE
from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet, FusionTrans, Encoder, \
    SelfAttention, CrossAttention, CrossAttentionWithMOE, CLUB, Projector, SelfAttentionAudio, SelfAttentionVision, \
    CrossAttentionWithFoldAttention, CrossDiffAttention


class TADCA(nn.Module):

    def __init__(self, hp):
        """Construct TADCA (Temporal-Aware Differential Cross-Modal Attention) model.
        Args:
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super().__init__()
        self.hp = hp

        self.add_va = hp.add_va
        hp.d_tout = hp.d_tin

        self.uni_text_enc = LanguageEmbeddingLayer(hp)  # BERT Encoder
        self.uni_visual_enc = RNNEncoder(  # 视频特征提取
            in_size=hp.d_vin,
            hidden_size=hp.d_vh,
            out_size=hp.d_vout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )
        self.uni_acoustic_enc = RNNEncoder(  # 音频特征提取
            in_size=hp.d_ain,
            hidden_size=hp.d_ah,
            out_size=hp.d_aout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )

        # For MI maximization   互信息最大化
        # Modality Mutual Information Lower Bound（MMILB）
        self.mi_tv = MMILB(
            x_size=hp.d_tout,
            y_size=hp.d_vout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        )

        self.mi_ta = MMILB(
            x_size=hp.d_tout,
            y_size=hp.d_aout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        )

        if hp.add_va:  # 一般是tv和ta   若va也要MMILB
            self.mi_va = MMILB(
                x_size=hp.d_vout,
                y_size=hp.d_aout,
                mid_activation=hp.mmilb_mid_activation,
                last_activation=hp.mmilb_last_activation
            )

        # CPC MI bound   d_prjh是什么？？？
        self.cpc_zt = CPC(
            x_size=hp.d_tout,  # to be predicted  各个模态特征提取后得到的维度
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )
        self.cpc_zv = CPC(
            x_size=hp.d_vout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )
        self.cpc_za = CPC(
            x_size=hp.d_aout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )

        self.uni_audio_encoder = SelfAttentionAudio(hp, d_in=hp.d_ain, d_model=hp.model_dim_self,
                                                   nhead=hp.num_heads_self,
                                                   dim_feedforward=4 * hp.model_dim_self, dropout=hp.attn_dropout_self,
                                                   num_layers=hp.num_layers_self)


        self.uni_vision_encoder = SelfAttentionVision(hp, d_in=hp.d_vin, d_model=hp.model_dim_self,
                                                    nhead=hp.num_heads_self,
                                                    dim_feedforward=4 * hp.model_dim_self, dropout=hp.attn_dropout_self,
                                                    num_layers=hp.num_layers_self)


        self.audio_recon_head = SubNet(in_size=hp.d_ain, hidden_size=hp.d_ain//2, n_class=hp.n_class, dropout=hp.dropout_prj, output_size=None)
        self.vision_recon_head = SubNet(in_size=hp.d_vin, hidden_size=hp.d_vin//2, n_class=hp.n_class, dropout=hp.dropout_prj, output_size=None)

        self.infonce_loss_va = InfoNCE(
            modal1_dim_in=hp.d_vin,
            modal2_dim_in=hp.d_ain,
            embed_dropout=hp.embed_dropout,
            temperature=0.1
        )

        self.infonce_loss_av = InfoNCE(
            modal1_dim_in=hp.d_ain,
            modal2_dim_in=hp.d_vin,
            embed_dropout=hp.embed_dropout,
            temperature=0.1
        )


        if self.hp.dataset == "sims":
            self.audio_classifer = SubNet(in_size=hp.d_aout, hidden_size=hp.d_aout*2,
                                    n_class=hp.n_class, dropout=hp.dropout_prj, output_size=None)  # [bs,seq_len,d_ain]
            self.vision_classifer = SubNet(in_size=hp.d_vout, hidden_size=hp.d_vout*2,
                                     n_class=hp.n_class, dropout=hp.dropout_prj, output_size=None)  # [bs,seq_len,d_vin]
            self.text_classifer = SubNet(in_size=hp.d_tin, hidden_size=hp.d_tin*2,
                                   n_class=hp.n_class, dropout=hp.dropout_prj, output_size=None)  # [bs,seq_len,d_tin]


        # 用MULT融合 每个模块的输出都是[bs,query_length,model_dim_cross]
        self.ta_cross_attn=CrossDiffAttention(hp,d_modal1=hp.d_tin,d_modal2=hp.d_ain,d_model=hp.model_dim_cross,nhead=hp.num_heads_cross,
                                          dim_feedforward=4*hp.model_dim_cross,dropout=hp.attn_dropout_cross,num_layers=hp.num_layers_cross,
                                              learnable_length=hp.learnable_length)
        self.tv_cross_attn=CrossDiffAttention(hp,d_modal1=hp.d_tin,d_modal2=hp.d_vin,d_model=hp.model_dim_cross,nhead=hp.num_heads_cross,
                                          dim_feedforward=4*hp.model_dim_cross,dropout=hp.attn_dropout_cross,num_layers=hp.num_layers_cross,
                                              learnable_length=hp.learnable_length)

        self.fusion_mlp_for_regression = SubNet(in_size=hp.model_dim_cross*2,hidden_size=hp.d_prjh,dropout=hp.dropout_prj,n_class=hp.n_class)

    def contrastive_loss(self,z_a, z_v, tau=0.1):
        # z_a, z_v: (seq_len, batch, dim) 或 (batch, dim) 取平均池化
        z_a = F.normalize(z_a.mean(dim=0), dim=-1)  # (batch, dim)
        z_v = F.normalize(z_v.mean(dim=0), dim=-1)

        sim_matrix = torch.matmul(z_a, z_v.T)  # (batch, batch)
        sim_matrix = sim_matrix / tau

        labels = torch.arange(z_a.size(0), device=z_a.device)
        loss_a2v = F.cross_entropy(sim_matrix, labels)
        loss_v2a = F.cross_entropy(sim_matrix.T, labels)
        return (loss_a2v + loss_v2a) / 2

    def compute_local_mask(self, unimodal_bottleneck, window_size=5, eps=1e-6):
        """
        unimodal_bottleneck: [seq_len, batch, dim]
        返回 mask: [batch, seq_len]  True表示被mask掉
        """
        seq_len, batch, dim = unimodal_bottleneck.shape

        # 归一化
        x_norm = F.normalize(unimodal_bottleneck, dim=-1)  # [seq_len, batch, dim]

        # 调整维度到 [batch, seq_len, dim] 方便 batch matmul
        x_norm_b = x_norm.permute(1, 0, 2)  # [batch, seq_len, dim]

        # 全局相似度矩阵: [batch, seq_len, seq_len]
        sim = torch.matmul(x_norm_b, x_norm_b.transpose(1, 2))  # [batch, seq_len, seq_len]

        # softmax归一化得到注意力权重，沿 seq_len 维度
        attn = F.softmax(sim, dim=1)  # [batch, seq_len, seq_len]

        mask_prob = torch.zeros(batch, seq_len)
        half_win = window_size // 2

        for t in range(seq_len):
            start = max(0, t - half_win)
            end = min(seq_len, t + half_win + 1)
            # ratio = 局部窗口内注意力和 / 全局注意力和
            local_sum = attn[:, start:end, t].sum(dim=1)  # [batch]
            global_sum = attn[:, :, t].sum(dim=1) + eps  # [batch]
            ratio = local_sum / global_sum
            mask_prob[:, t] = ratio

        mask = torch.bernoulli(mask_prob).bool()
        return mask  # [batch, seq_len]

    def gen_mask(self, a, length=None):
        if length is None:
            msk_tmp = torch.sum(a, dim=-1)
            # 特征全为0的时刻加mask
            mask = (msk_tmp == 0)
            return mask
        else:
            b = a.shape[0]
            l = a.shape[1]
            msk = torch.ones((b, l))
            x = []
            y = []
            for i in range(b):
                for j in range(length[i], l):
                    x.append(i)
                    y.append(j)
            msk[x, y] = 0
            return (msk == 0)

    def normalize(self,*xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]

    def transpose(self,x):
        return x.transpose(-2, -1)


    def kl_divergence(self,p,q):
        # 计算方差
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        loss = F.kl_div(q.log(), p, reduction='batchmean')
        return loss

    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None,
                mem=None,v_mask=None,a_mask=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        sentences: torch.Size([0, 32])
        a: torch.Size([134, 32, 5])
        v: torch.Size([161, 32, 20])
        For Bert input, the length of text is "seq_len + 2"
        """
        with torch.no_grad():
            maskT = (bert_sent_mask == 0)
            maskV = self.gen_mask(visual.transpose(0,1),v_len)
            maskA = self.gen_mask(acoustic.transpose(0,1),a_len)

        enc_word= self.uni_text_enc(sentences, bert_sent, bert_sent_type,bert_sent_mask)  # 32*50*768 (batch_size, seq_len, emb_size)
        text_trans = enc_word.transpose(0, 1)  # torch.Size([50, 32, 768]) (seq_len, batch_size,emb_size)

        acoustic_enc = self.uni_audio_encoder(acoustic)  # [seq_len,bs,dim] 自注意力
        visual_enc = self.uni_vision_encoder(visual)  # [seq_len,bs,dim] 自注意力

        audio_bottleneck = acoustic_enc
        vision_bottleneck = visual_enc

        if self.training:

            local_maskA=self.compute_local_mask(audio_bottleneck, window_size=self.hp.local_window_size_audio, eps=1e-6)
            local_maskV=self.compute_local_mask(vision_bottleneck,window_size=self.hp.local_window_size_visual, eps=1e-6)

            final_audio_mask = maskA | local_maskA
            final_vision_mask = maskV | local_maskV

        else:
            final_audio_mask = maskA
            final_vision_mask = maskV

        # 2. 跨模态注意力部分
        cross_tv = self.tv_cross_attn(text_trans, visual_enc,Tmask=maskT,Vmask=final_vision_mask).mean(dim=0)
        cross_ta = self.ta_cross_attn(text_trans, acoustic_enc,Tmask=maskT,Amask=final_audio_mask).mean(dim=0)

        fusion, preds = self.fusion_mlp_for_regression(torch.cat([cross_ta, cross_tv], dim=1))  # 32*128,32*1


        if self.training:
            text = enc_word[:,0,:] # 32*768 (batch_size, emb_size)
            acoustic = self.uni_acoustic_enc(acoustic, a_len)  # 32*16
            visual = self.uni_visual_enc(visual, v_len)  # 32*16

            if self.hp.dataset == "sims":
                _, acoustic_pred = self.audio_classifer(acoustic)
                _, visual_pred = self.vision_classifer(visual)
                _, text_pred = self.text_classifer(text)
            else:
                acoustic_pred,visual_pred,text_pred =None,None,None

            if y is not None:
                lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual, labels=y, mem=mem['tv'])
                lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic, labels=y, mem=mem['ta'])
                # for ablation use
                if self.add_va:
                    lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic, labels=y, mem=mem['va'])
            else:  # 默认进这
                lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual)  # mi_tv 模态互信息
                # lld_tv:-2.1866  tv_pn:{'pos': None, 'neg': None}  H_tv:0.0
                lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic)
                if self.add_va:
                    lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic)

            nce_t = self.cpc_zt(text, fusion)  # 3.4660
            nce_v = self.cpc_zv(visual, fusion)  # 3.4625
            nce_a = self.cpc_za(acoustic, fusion)  # 3.4933

            nce = nce_t + nce_v + nce_a  # 10.4218  CPC loss

            pn_dic = {'tv': tv_pn, 'ta': ta_pn, 'va': va_pn if self.add_va else None}
            # {'tv': {'pos': None, 'neg': None}, 'ta': {'pos': None, 'neg': None}, 'va': None}
            lld = lld_tv + lld_ta + (lld_va if self.add_va else 0.0)  # -5.8927
            H = H_tv + H_ta + (H_va if self.add_va else 0.0)
        if self.training:
            return lld, nce, preds, pn_dic, H,text_pred,visual_pred,acoustic_pred
        else:
            return None,None, preds, None, None,None,None,None

if __name__=="__main__":
    net=Encoder(4, 8, 2,32,0.1,'relu',2)
    data=torch.randn(30,32,4)
    data_mask=pad_sequence([torch.zeros(torch.FloatTensor(sample).size(0)) for sample in data])
    data_mask[:,4:].fill_(float(1.0))
    output=net(data,data_mask.transpose(1,0))
    print(data_mask,data)