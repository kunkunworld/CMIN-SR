"""Non-neural control with existing log-volatility summary features."""
import numpy as np
import torch
from mrw_inverse.models.logvol_covariance import LogVolCovarianceBranch


def logvol_features(x):
    branch=LogVolCovarianceBranch().eval()
    pieces=[]
    with torch.no_grad():
        for s in range(0,len(x),64):
            d=branch(x[s:s+64].cpu())
            pieces.append(torch.cat([d['logvol_covariance_curve'],d['logvol_covariance_slope']],dim=1).numpy())
    return np.concatenate(pieces)
