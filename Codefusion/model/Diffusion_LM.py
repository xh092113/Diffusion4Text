import torch
from torch import nn
import numpy as np
import math
from transformers import (
    # BertModel,
    # BertConfig,
    T5EncoderModel,
    AutoConfig,
)
from model.CrossAttentionTransformers import BasicTransformerBlock

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class CrossAttention_Diffusion_LM(nn.Module):
    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            dropout=0,
            config=None,
            # config_name='bert-base-uncased',
            config_name="Salesforce/codet5p-220m-bimodal",
            vocab_size=None,
            init_pretrained=True,
            logits_mode=1,
            token_emb_type='pretrain',
            fix_encoder=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.init_pretrained = init_pretrained
        self.token_emb_type = token_emb_type
        self.fix_encoder = fix_encoder

        # cfg = BertConfig.from_pretrained(config_name)
        # cfg.num_hidden_layers = 6
        # self.passage_encoder = BertModel.from_pretrained(config_name, config=cfg)
        # # self.passage_encoder = BertModel.from_pretrained(
        # #     "/colab_space/Lin0/PROD/KDexp/pretrain_model/bert-base-uncased", config=cfg)
        # config = BertConfig.from_pretrained(config_name)
        config = AutoConfig.from_pretrained(config_name)
        config.hidden_dropout_prob = self.dropout
        # print(config)
        self.passage_encoder=T5EncoderModel.from_pretrained(
            "Salesforce/codet5p-220m-bimodal",
            trust_remote_code=True,
        )
        self.passage_encoder.requires_grad=False

        # trainable embedding layer
        self.word_embedding = nn.Embedding(vocab_size, self.in_channels)
        # position embedding
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if self.logits_mode == 2:
            # self.lm_head = nn.Linear(self.in_channels, vocab_size, bias=False)
            self.lm_head = nn.Linear(self.in_channels, vocab_size, bias=True)
        else:
            self.lm_head = nn.Linear(self.in_channels, vocab_size)

        # share weight between lm_head and word_embedding
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight
        #待办!!!!!!!!!!!!!!!!!!!!!!!!!分类头嘛, 倒是没问题, 但是我们需要在这下面加一个6层的decoder
        #还要配以相应的loss
        config.num_hidden_layers = 6
        self.decoder = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.hidden_size // config.num_attention_heads,
                    dropout=config.hidden_dropout_prob,
                    cross_attention_dim=config.hidden_size,
                    activation_fn="geglu",
                )
                for d in range(config.num_hidden_layers)
            ]
        )

        # self.word_embedding = nn.Embedding(vocab_size, self.in_channels)
        # self.lm_head = nn.Linear(self.in_channels, vocab_size)
        # with th.no_grad():
        #     self.lm_head.weight = self.word_embedding.weight

        # time embedding layer
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, config.hidden_size),
        )

        # position embedding
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # # label embedding
        # if self.num_classes is not None:
        #     self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # input transform
        self.input_up_proj = nn.Sequential(
            nn.Linear(in_channels, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        # Dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        config.num_hidden_layers = 10
        # define cross attention transformer block(6 layer)
        # 现在改成10了
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.hidden_size // config.num_attention_heads,
                    dropout=config.hidden_dropout_prob,
                    cross_attention_dim=config.hidden_size,
                    activation_fn="geglu",
                )
                for d in range(config.num_hidden_layers)
            ]
        )

        # output transform
        self.output_down_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, out_channels)
        )

    def get_embeds(self, input_ids):
        # if torch.max(input_ids) >= 31735:
        #     print(input_ids.shape)
        #     print(torch.max(input_ids))
        #     _ = input()
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = torch.sqrt(torch.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()

            return scores
        else:
            raise NotImplementedError
            
    def encode(self, src_input_ids, src_attention_mask):
        if self.fix_encoder:
            with torch.no_grad():
                out = self.passage_encoder(input_ids=src_input_ids,
                                                 attention_mask=src_attention_mask)
                passage_hidden = out.last_hidden_state
        else:
            out = self.passage_encoder(input_ids=src_input_ids,
                                       attention_mask=src_attention_mask)
            passage_hidden = out.last_hidden_state + 0 * out.pooler_output.unsqueeze(1)
        return passage_hidden
            
    def decode(self, hidden_states, passage_hidden):
        for block in self.decoder:
            hidden_states = block(hidden_states, passage_hidden)
        return hidden_states

    def forward(self, x, timesteps, passage_hidden):

        # prepare embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length]
        # print(emb_x.shape, emb.shape, self.position_embeddings)
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1)
        hidden_states = self.dropout(self.LayerNorm(emb_inputs))
        # encode embedding
        # print(emb_inputs.shape, attention_mask.shape)
        
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, passage_hidden)

        h = self.output_down_proj(hidden_states)
        h = h.type(x.dtype)
        return h