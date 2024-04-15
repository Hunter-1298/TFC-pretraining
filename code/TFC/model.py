from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

""" Contrastive Encoder for Time """
class TFC(nn.Module):
    def __init__(self, configs):
        super(TFC, self).__init__()
        # TransformerEncoderLayer(d_model: int - number of expected features in the input,dim_feedforward: int - dimension of feedforward network nhead: int - number of head in multiheadattention)
        encoder_layers_t = TransformerEncoderLayer(96, dim_feedforward=2*96, nhead=2)
        # Stacks x number of transformerEncoderLayers together
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)
        # Onlyn need projected if we are going to use joint TF space
        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x_in_t):#, x_in_f):
        """Use Transformer"""
        x = self.transformer_encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)
        return h_time

        """ No need for freq or cross space yet
        #Cross-space projector
        z_time = self.projector_t(h_time)

        #Frequency-based contrastive encoder
        f = self.transformer_encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        #Cross-space projector
        z_freq = self.projector_f(h_freq)
        """
        return h_time #, z_time, h_freq, z_freq

# We are not going to fine tune, just pretrain so comment out
"""
#Downstream classifier only used in finetuning
class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2*128, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred
"""