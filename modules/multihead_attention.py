import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys
import numpy as np


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


# Code adapted from the fairseq repo.

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim  # 30
        self.num_heads = num_heads  # 5
        self.attn_dropout = attn_dropout  # 0
        self.head_dim = embed_dim // num_heads  # 30 // 6 =5
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
        self.scaling = self.head_dim ** -0.5  # ** ：乘方（指数）根号dk 收缩因子

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))  # 使得in_proj_weight变得可优化
        self.register_parameter('in_proj_bias', None)
        # in_proj_bias 的意思就是一开始的线性变换的偏置。
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn  # FALSE

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)


    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()  # false  # data_ptr():返回tensor第一个元素的地址
        kv_same = key.data_ptr() == value.data_ptr()  # false

        # key, value：500*8*30
        tgt_len, bsz, embed_dim = query.size()  # 50，8，30
        # 以下断言都是为了确认参数合法
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:  # false
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:  # false
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:  # 这里
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling  # 根号dk

        if self.bias_k is not None:  # 不进
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # 32*5,50, 30/5
        # contiguous() 返回开辟了一块新的存放q的连续内存，并且改变该值会改变原值
        # head_dim = embed_dim(30) // num_heads(5)
        # view(50, 8*5, 6)  -> 40*50*6
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # 32*5,147, 30/5
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # 32*5,147, 30/5

        src_len = k.size(1)  # 50

        if self.add_zero_attn:  # 没进
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))  # 32*5,l1,l2
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:  # 进
            try:
                # attn_weights += attn_mask.unsqueeze(0)  # attn_mask：50*500
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads, 1, query.shape[0], 1)
                attn_weights = attn_weights.view(self.num_heads, query.shape[1], query.shape[0], -1).masked_fill(
                    attn_mask.cuda(), -np.inf)
                attn_weights = attn_weights.view(self.num_heads * query.shape[1], query.shape[0], -1)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)



        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)  # 32*5,50,30/5 attn_weights: 40*50*500  v:40*500*6
        # bmm: bnm 和 bmp 得到 bnp   -> 40*50*6   具体运算看：https://blog.csdn.net/weixin_45573525/article/details/108143684
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)  # 拼回去，50，32，30
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)  # 32,5,50,147
        attn_weights = attn_weights.sum(dim=1) / self.num_heads  # 对注意力分数的5个头求均值
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        # 以 q = self.in_proj_q(query)为例
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):  # input: query ,end: 30
        # 以 self._in_proj(query, end=self.embed_dim, **kwargs) 为例
        weight = kwargs.get('weight', self.in_proj_weight)  # 30*30
        bias = kwargs.get('bias', self.in_proj_bias)  # shape=90, size:1
        weight = weight[start:end, :]  # 30*30
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)




class KernelMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, attn_drop=0., kernel_sizes=[3,2,1,1,1]):
        super().__init__()
        assert len(kernel_sizes) == num_heads, f"kernel_sizes长度 ({len(kernel_sizes)}) 必须等于 num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.kernel_sizes = kernel_sizes  # 每个head对应的kernel size
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV投影矩阵
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query,key,value,attn_mask=None):
        """
        输入:
            x: (seq_len, batch_size, dim)
        """
        tgt_len, batch_size, _ = query.shape
        src_len,_,_ = key.shape
        assert key.size()==value.size()
        q=query.contiguous().view(tgt_len, batch_size, self.num_heads, self.head_dim)
        k=key.contiguous().view(src_len, batch_size, self.num_heads, self.head_dim)
        v=value.contiguous().view(src_len, batch_size, self.num_heads, self.head_dim)

        # 存储每个head的注意力输出
        head_outputs = []
        attn_mask_=attn_mask
        # 对每个head分别计算注意力
        for head_idx in range(self.num_heads):
            kernel_size = self.kernel_sizes[head_idx]
            q_head = q[:, :, head_idx]
            k_head = k[:, :, head_idx]
            v_head = v[:, :, head_idx]

            # 使用切片获取完整的块
            q_chunks = q_head.transpose(1,0)
            k_chunks = k_head[::kernel_size].contiguous().view(-1,batch_size, self.head_dim).transpose(1,0)
            v_chunks = v_head[::kernel_size].contiguous().view(-1, batch_size, self.head_dim).transpose(1,0)

            attn_mask=attn_mask_[:,::kernel_size]

            src_chunk_len=k_chunks.size(1)
            # 计算注意力分数
            attn = torch.matmul(q_chunks, k_chunks.transpose(-2, -1)) * self.scale
            assert list(attn.size()) == [batch_size, tgt_len, src_chunk_len]

            if attn_mask is not None:  # 进
                try:
                    # attn_weights += attn_mask.unsqueeze(0)  # attn_mask：50*500
                    attn_mask = attn_mask.unsqueeze(1).unsqueeze(0).repeat(1, 1, query.shape[0], 1)
                    attn = attn.view(1, query.shape[1], query.shape[0], -1).masked_fill(
                        attn_mask.cuda(), -np.inf)
                    attn = attn.view(query.shape[1], query.shape[0], -1)
                except:
                    print(attn.shape)
                    print(attn_mask.unsqueeze(0).shape)
                    assert False
            attn= F.softmax(attn.float(), dim=-1).type_as(attn)
            attn = self.attn_drop(attn)

            # 应用注意力
            out_chunks = torch.matmul(attn, v_chunks)  # (num_chunks, kernel_size, batch_size, head_dim)

            # 重塑回序列形式
            out = out_chunks.reshape(tgt_len, batch_size, self.head_dim)
            head_outputs.append(out)

        # 合并所有head的输出
        out = torch.cat(head_outputs, dim=-1)  # (seq_len, batch_size, dim)

        # 最终投影
        out = self.proj(out)
        return out,None


class MultiheadDiffAttention(nn.Module):
    """
    差分多头注意力 (DiffMHA)
    - 参数接口对齐标准 MultiheadAttention
    - Dropout/mask/输出接口保持一致
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False,
                 depth=0, num_kv_heads=None,learnable_length=20):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = embed_dim // num_heads // 2
        assert self.head_dim > 0, "embed_dim 必须足够大才能除以 2*num_heads"

        self.scaling = self.head_dim ** -0.5
        self.attn_dropout = attn_dropout

        # Q/K/V 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # bias_k, bias_v（与标准 MultiheadAttention 保持一致）
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.learnable_length = learnable_length
        # λ 初始化
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))

        # RMSNorm
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """
        输入: (T, B, C)
        输出: attn, attn_weights
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)

        # Q, K, V
        q = self.q_proj(query.transpose(0, 1))   # (B, T, C)
        k = self.k_proj(key.transpose(0, 1))     # (B, S, C//n_rep)
        v = self.v_proj(value.transpose(0, 1))   # (B, S, C//n_rep)

        # 如果 add_bias_kv
        if self.bias_k is not None and self.bias_v is not None:
            k = torch.cat([k, self.bias_k.expand(bsz, -1, -1)], dim=1)
            v = torch.cat([v, self.bias_v.expand(bsz, -1, -1)], dim=1)
            src_len += 1
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        # reshape
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim).transpose(1, 2)  # (B, 2H, T, d)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, 2Hk, S, d)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim).transpose(1, 2)  # (B, Hk, S, 2d)

        # KV repeat
        k = repeat_kv(k, self.n_rep)  # (B, H, S, d)
        v = repeat_kv(v, self.n_rep)  # (B, H, S, 2d)

        # add_zero_attn
        if self.add_zero_attn:
            k = torch.cat([k, k.new_zeros((k.size(0), k.size(1), 1, k.size(-1)))], dim=2)
            v = torch.cat([v, v.new_zeros((v.size(0), v.size(1), 1, v.size(-1)))], dim=2)
            src_len += 1
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        # Scaled dot-product
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))  # (B, 2H, T, S)

        # -------- Mask 逻辑完全照搬原始 MultiheadAttention --------
        if attn_mask is not None:
            try:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(0).unsqueeze(2).repeat(self.num_heads, 1,2, tgt_len, 1)
                attn_weights = attn_weights.view(self.num_heads, bsz, 2, tgt_len, src_len)
                attn_weights = attn_weights.masked_fill(attn_mask.bool().cuda(), -np.inf)
                attn_weights = attn_weights.view(bsz, 2 * self.num_heads, tgt_len, src_len)
            except Exception as e:
                print("attn_weights.shape:", attn_weights.shape)
                print("attn_mask.shape:", attn_mask.shape)
                raise e
        # --------------------------------------------------------

        # softmax
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)

        # dropout
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        # 差分 λ
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # 差分操作
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        # 1. reshape lambda_full 到 (N,C,L) 格式，N=C=1
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        # 注意力聚合
        attn = torch.matmul(attn_weights, v)  # (B, H, T, 2d)

        # RMSNorm + 缩放
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)

        # 还原形状 (T, B, C)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        attn = self.out_proj(attn).transpose(0, 1)

        return attn, attn_weights

def gen_mask(a, length=None):
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



if __name__=="__main__":
    query = torch.rand([50,16,30]).cuda()   # seq_len, batch, embed_dim
    key = torch.rand([195,16,30]).cuda()
    value = torch.rand([195,16,30]).cuda()
    attn_mask = gen_mask(key.transpose(1,0),length=[195]*16)
    m = KernelMultiHeadAttention(embed_dim=30,num_heads=5).cuda()
    y = m(query,key,value,attn_mask)
    print(y.shape)