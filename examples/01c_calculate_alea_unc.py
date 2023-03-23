import pandas as pd

from ILC_analyzer.nigtools import NigDist

if __name__ == "__main__":
    path_data = '../data/ILC_data.xlsx'

    # Load data from excel file
    df = pd.read_excel(path_data)
    x_i = df['Avg'].values
    sigma_i = df['std'].values

    # Create an instance of NigDist to calculate parameters of normal-inverse-gamma distribution
    ilc = NigDist(x=x_i, sigma=sigma_i,k=1)

    # Print estimated parameters of NIG distribution
    print(f"Estimated parameters of NIG distribution: mu={ilc.mu:.4f}, lambda={ilc.lambda_:.4f}, alpha={ilc.alpha:.4f}, beta={ilc.beta:.4f}")

    # Calculate aleatoric uncertainty E[s^2]
    unc_ale = ilc.unc_aleatoric
    print(f'Calculated aleatoric uncertainty: {unc_ale:.3f}')

    # Calculate epistemic uncertainty var[x]
    unc_epist = ilc.unc_epistemic
    print(f'Calculated epistemic uncertainty: {unc_epist:.3f}')

    # Plot the data with the calculated uncertainty
    ilc.plot_x_with_uncertainty()
