import numpy as np
import pandas as pd
from ILC_analyzer.nigtools import (
    estimate_nig_params,
    aleatoric_uncertainy_from_nig,
    epistemic_uncertainy_from_nig,
)
from ILC_analyzer.nigtools import NigDist
from ILC_analyzer.plotting import plot_x_with_uncertainty

if __name__ == "__main__":
    path_data = '../data/ILC_data.xlsx'

    # Load data from excel file
    df = pd.read_excel(path_data)
    x_i = np.array(df['Avg'].values)
    sigma_i = np.array(df['std'].values)

    # Calculate parameters of normal-inverse-gamma distribution
    dist=NigDist(x=x_i,sigma=sigma_i)
    mu, lambda_, alpha, beta = dist.estimate_nig_params()


    print(f"Estimated parameters of NIG distribution: mu={mu:.4f}, lambda={lambda_:.4f}, alpha={alpha:.4f}, beta={beta:.4f}")

    # Calculate aleatoric uncertainty E[s^2]
    #unc_ale = aleatoric_uncertainy_from_nig(alpha, beta)
    unc_ale=dist.unc_aleatoric
    print(f'Calculated aleatoric uncertainty: {unc_ale:.3f}')

    # Calculate epistemic uncertainty var[x]
    unc_epist = dist.unc_epistemic
    print(f'Calculated epistemic uncertainty: {unc_epist:.3f}')

    #plot_x_with_uncertainty(x_i, sigma_i, mu, unc_alea=unc_ale)
