import torch
from torch import nn

import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

# import transformers

from Models.BolT.bolTransformerBlock import BolTransformerBlock

class BolT(nn.Module):
    def __init__(self, hyperParams, details):
        super().__init__()

        dim = hyperParams.dim
        self.hyperParams = hyperParams

        self.inputNorm = nn.LayerNorm(dim)
        self.clsToken = nn.Parameter(torch.zeros(1, 1, dim))

        self.blocks = nn.ModuleList()
        shiftSize = int(hyperParams.windowSize * hyperParams.shiftCoeff)
        self.shiftSize = shiftSize
        self.receptiveSizes = []

        for i in range(hyperParams.nOfLayers):
            if hyperParams.focalRule == "expand":
                receptiveSize = hyperParams.windowSize + math.ceil(hyperParams.windowSize * 2 * i * hyperParams.fringeCoeff * (1 - hyperParams.shiftCoeff))
            elif hyperParams.focalRule == "fixed":
                receptiveSize = hyperParams.windowSize + math.ceil(hyperParams.windowSize * 2 * 1 * hyperParams.fringeCoeff * (1 - hyperParams.shiftCoeff))

            self.receptiveSizes.append(receptiveSize)

            self.blocks.append(BolTransformerBlock(
                dim=dim,
                numHeads=hyperParams.numHeads,
                headDim=hyperParams.headDim,
                windowSize=hyperParams.windowSize,
                receptiveSize=receptiveSize,
                shiftSize=shiftSize,
                mlpRatio=hyperParams.mlpRatio,
                attentionBias=hyperParams.attentionBias,
                drop=hyperParams.drop,
                attnDrop=hyperParams.attnDrop
            ))

        self.encoder_postNorm = nn.LayerNorm(dim)

        # remove classification head
        # self.classifierHead = nn.Linear(dim, nOfClasses)

        
        self.initializeWeights()

        # for analysis/debugging
        self.tokens = []
        self.last_numberOfWindows = None

    def initializeWeights(self):
        torch.nn.init.normal_(self.clsToken, std=1.0)

    def forward(self, roiSignals, analysis=False):
        """
        Input:
            roiSignals: (batchSize, N, T)
        Output:
            embedding: (batchSize, embedding_dim)
        """
        roiSignals = roiSignals.permute((0, 2, 1))  # -> (B, T, C)
        

        batchSize, T = roiSignals.shape[0], roiSignals.shape[1]
        nW = (T - self.hyperParams.windowSize) // self.shiftSize + 1
        cls = self.clsToken.repeat(batchSize, nW, 1)

        self.last_numberOfWindows = nW

        if analysis:
            self.tokens.append(torch.cat([cls, roiSignals], dim=1))

        for block in self.blocks:
            roiSignals, cls = block(roiSignals, cls, analysis)
            if analysis:
                self.tokens.append(torch.cat([cls, roiSignals], dim=1))

        cls = self.encoder_postNorm(cls)  # (B, nW, D)

        # Pool cls tokens to get final embedding
        embedding = cls.mean(dim=1)  # (B, D)

        # Normalize for cosine similarity
        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding, cls  # Optionally: only return embedding if cls isn't needed


class ShallowClassifier(nn.Module):
    def __init__(self, hyperParams, details):
        super().__init__()
        self.hyperParams = hyperParams
        self.details = details
        self.num_layers = hyperParams.nOfLayers
        self.classifierHead = torch.nn.Sequential(
            torch.nn.Linear(hyperParams.dim, hyperParams.dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hyperParams.dim//2, hyperParams.dim//4),
            torch.nn.ReLU(),
            torch.nn.Linear(hyperParams.dim//4, details.nOfClasses)
        )
        self.initializeWeights()



    def initializeWeights(self):
        torch.nn.init.xavier_uniform_(self.classifierHead.weight)  # Xavier initialization for weights
        torch.nn.init.zeros_(self.classifierHead.bias)  # Zero initialization for bias
    
    def forward(self, x):
        x = self.classifierHead(x)
        return x