# src/portrisk/risk.py

import numpy as np
import scipy.stats as stats
from typing import Optional


def cornish_fisher_var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR) using the Cornish-Fisher expansion.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of portfolio returns.
    alpha : float
        Significance level (e.g., 0.05 for 5% VaR).
        
    Returns
    -------
    float
        Cornish-Fisher VaR estimate (negative value represents loss).
    """
    mean = np.mean(returns)
    sigma = np.std(returns)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns, fisher=True)

    # Standard normal quantile
    z = stats.norm.ppf(alpha)

    # Cornish-Fisher adjustment
    z_cf = (z +
            (1/6)*(z**2 - 1)*skew +
            (1/24)*(z**3 - 3*z)*(kurt) -
            (1/36)*(2*z**3 - 5*z)*(skew**2))
    
    var = -(mean + z_cf * sigma)
    return var


def pot_var(returns: np.ndarray,
            threshold: Optional[float] = None,
            alpha: float = 0.05) -> float:
    """
    Calculate VaR using Peaks-Over-Threshold (POT) method with GPD.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of portfolio returns.
    threshold : float, optional
        Threshold for excesses. If None, 95th percentile is used.
    alpha : float
        Significance level (e.g., 0.05 for 5% VaR).
    
    Returns
    -------
    float
        POT VaR estimate (negative value represents loss).
    """
    # Use negative returns for losses
    losses = -returns
    
    # Set threshold if not provided
    if threshold is None:
        threshold = np.percentile(losses, 95)
    
    # Exceedances over threshold
    excesses = losses[losses > threshold] - threshold
    n_excess = len(excesses)
    n_total = len(losses)
    
    # Fit Generalized Pareto Distribution
    c, loc, scale = stats.genpareto.fit(excesses, floc=0)
    
    # POT VaR formula
    prob_exceed = n_excess / n_total
    quantile = stats.genpareto.ppf(
        (alpha - (1 - prob_exceed)) / prob_exceed,
        c, loc=0, scale=scale
    )
    var = -(threshold + quantile)
    return var
