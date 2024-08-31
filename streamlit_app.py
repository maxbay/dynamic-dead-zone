import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle
import polars as pl
from streamlit_searchbox import st_searchbox
from scipy.stats import multivariate_normal
from typing import Any, List, Tuple


from numpy import linalg as la
from matplotlib.patches import Ellipse

def get_spectral_colors(n,pal='Spectral'):
    cmap = plt.get_cmap(pal)
    colors = [cmap(i / n) for i in range(n)]
    return colors

def plot_confidence_ellipse(mu, cov, alph, ax, clabel=None, label_bg='white', clabel_size=10, **kwargs):
    """Display a confidence ellipse of a bivariate normal distribution
    
    Arguments:
        mu {array-like of shape (2,)} -- mean of the distribution
        cov {array-like of shape(2,2)} -- covariance matrix
        alph {float btw 0 and 1} -- level of confidence
        ax {plt.Axes} -- axes on which to display the ellipse
        clabel {str} -- label to add to ellipse (default: {None})
        label_bg {str} -- background of clabel's textbox
        clabel_size {int} -- font size of the clabel (default: {10})

        kwargs -- other arguments given to class Ellipse
    """
    c = -2 * np.log(1 - alph)  # quantile at alpha of the chi_squarred distr. with df = 2
    Lambda, Q = la.eig(cov)  # eigenvalues and eigenvectors (col. by col.)
    
    ## Compute the attributes of the ellipse
    width, heigth = 2 * np.sqrt(c * Lambda)
    # compute the value of the angle theta (in degree)
    theta = 180 * np.arctan(Q[1,0] / Q[0,0]) / np.pi if cov[1,0] else 0
        
    ## Create the ellipse
    if 'fc' not in kwargs.keys():
        kwargs['fc'] = 'None'
    level_line = Ellipse(mu, width, heigth, angle=theta, **kwargs)
    
    ## Display a label 'clabel' on the ellipse
    if clabel:
        col = kwargs['ec'] if 'ec' in kwargs.keys() and kwargs['ec'] != 'None' else 'black'  # color of the text
        pos = Q[:,1] * np.sqrt(c * Lambda[1]) + mu  # position along the heigth
        
        ax.text(*pos, clabel, color='black',
           rotation=theta, ha='center', va='center', rotation_mode='anchor', # rotation
           fontsize=clabel_size,  # set the font size
           bbox=dict(boxstyle='round',ec='None',fc=label_bg, alpha=1)) # white box
        
    return ax.add_patch(level_line)


def search_players(searchterm: str) -> List[str]:
    return [player for player in players if searchterm.lower() in player.lower()]

import numpy as np
from scipy.linalg import sqrtm

def wasserstein_distance(mu1, cov1, mu2, cov2):
    """
    Compute the 2-Wasserstein distance between two multivariate normal distributions.
    
    Parameters:
    mu1, mu2: Mean vectors of the two distributions (numpy arrays).
    cov1, cov2: Covariance matrices of the two distributions (numpy arrays).
    
    Returns:
    2-Wasserstein distance (float).
    """
    # Ensure inputs are numpy arrays
    mu1, mu2 = np.array(mu1), np.array(mu2)
    cov1, cov2 = np.array(cov1), np.array(cov2)
    
    # Compute the difference in means
    mean_diff = np.linalg.norm(mu1 - mu2)**2
    
    # Compute the square root of the covariance matrix product
    sqrt_cov_prod = sqrtm(np.dot(sqrtm(cov1), np.dot(cov2, sqrtm(cov1))))
    
    # Wasserstein distance formula for multivariate normal distributions
    wasserstein_dist = np.sqrt(mean_diff + np.trace(cov1 + cov2 - 2 * sqrt_cov_prod))
    
    return wasserstein_dist


# App Title and Description
st.title("Expected vs Observed FF Shape")

st.write("This app visualizes the shape of a pitcher's four-seam fastball (FF) relative to the expected shape given the picher's release position. You can select a pitcher and visualize the shape of their fastball for different years. A more unlikely shape is likely to be more surprising to the batter and more effective at limiting damage.")
st.text('')
st.write('I use release-direction-relative acceleration as the operant metric for shape because it is independent of time under a constant acceleration model, and I did not want time to plate as a source of variance. You can think of the acceleration components as, basically, induced vertical break and horizontal break.')
st.markdown('####')
# get player params
dict_path = './data/params.pkl'
with open(dict_path, 'rb') as f:
    player_params = pickle.load(f)

# get player df
df_path = './data/considered_players.csv'
considered_playeryears = pl.read_csv(df_path)

# player Selection with Autocomplete
players = considered_playeryears['player_name'].unique().to_list()

selected_player = st_searchbox(search_players, label="Select a Pitcher",default='Cole, Gerrit')

years = considered_playeryears.filter(pl.col('player_name') == selected_player)['game_year'].unique().to_list()

# Data Selection Options
st.write(f"Select the years you want to visualize:")

cols = st.columns(len(years))

checkboxes = {}
for i, year in enumerate(years):
    checkboxes[year] = cols[i].checkbox(str(year), value=(year == max(years)))

selected_years = [year for year in years if checkboxes[year]]

f, ax = plt.subplots(1, 1, figsize=(10, 10))

if any([checkboxes[year] for year in years]):

    latest_year = max(selected_years)

    colors = get_spectral_colors(5, pal = 'Set1')
    #######

    pitcher_name = selected_player
    pitcher = considered_playeryears.filter(pl.col('player_name') == pitcher_name)['pitcher'].to_numpy()[0]

    # Parameters
    #expected
    mu_expected = player_params[pitcher][latest_year]['mu_expected']
    sig_expected = player_params[pitcher][latest_year]['sig_expected']

    #arm_angle = considered_playeryears.filter((pl.col('player_name') == pitcher_name) & (pl.col('game_year') == latest_year))['arm_angle'].to_numpy()[0]
    
    clabel_size = 7

    # Define grid for contour plot
    x = np.linspace(-5, 20, 100)
    y = np.linspace(-5, 25, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Compute the bivariate normal distribution
    rv = multivariate_normal(mu_expected, sig_expected)
    Z = rv.pdf(pos)

    # Plot the spectral gradient without contour lines
    contour = ax.contourf(X, Y, Z, cmap='viridis', alpha=0.75, levels=5, antialiased=True)

    for i, year in enumerate(selected_years):
        #actual
        mu_actual = player_params[pitcher][year]['mu_actual']
        sig_actual = player_params[pitcher][year]['sig_actual']

        # Plot the other elements
        plot_confidence_ellipse(mu_actual, sig_actual, 0.5, ax, ec=colors[year - 2020], clabel=f'{year}', clabel_size=clabel_size)
        ax.scatter(*mu_actual, c=colors[year - 2020])

    ax.set_title(f"Pitch Shape Relative to Expected Given Arm Angle\n{pitcher_name} | FF", fontsize=18)


    ax.set_xlim(-5, 20)

    ax.set_ylim(-5, 25)
    ax.set_xlabel("horizontal acceleration ($ft/s^2$) | relative to release direction", fontsize=13)
    ax.set_ylabel("vertical acceleration ($ft/s^2$) | relative to release direction", fontsize=13)

    #arm_angle_from_top = 90 - arm_angle

    # Add arm angle as a grey dashed line
    #arm_angle_rad = np.deg2rad(arm_angle_from_top)  # Convert to radians if necessary
    #line_length = 30  # Length of the line
    #x_line = [0, line_length * np.cos(arm_angle_rad)]
    #y_line = [0, line_length * np.sin(arm_angle_rad)]
    #ax.plot(x_line, y_line, linestyle='--', color='#444444')

    # Add text just above the arm angle line
    #mid_x = line_length * np.cos(arm_angle_rad) * .2
    #mid_y = line_length * np.sin(arm_angle_rad)  * .2
    #text_offset = 1  # Offset to place the text just above the line
    #ax.text(mid_x, mid_y + text_offset, f'Arm Angle: {round(arm_angle,1)}°', color='#444444', fontsize=12, rotation=arm_angle_from_top, rotation_mode='anchor', ha='center', va='center')


    ax.axhline(0, color='black', linestyle='--')
    ax.axvline(0, color='black', linestyle='--')
    ax.set_aspect('equal')
    # Display the Plot
    st.pyplot(f)

# Explanation with LaTeX

st.write('\n\n\n')
st.write("### Explainer... it's just a conditional MVN")

st.write('')
st.write(r'''

    Population release position $(x,y,z)$ and pitch acceleration $(a_x,a_z)$ can be jointly modeled as 
    a single $5-dimensional$ multivariate normal distribution $X$ fit from population sample where
            
    $$
    X \sim  \mathcal{N}(\mu, \Sigma)
    $$
            
    To learn conditional distribution of acceleration given a release position, $X$ is partitioned into release position and acceleration components 

    $$
    \mathbf{x} =
    \begin{bmatrix}
    \mathbf{x}_{acc} \\
    \mathbf{x}_{rel}
    \end{bmatrix}

    $$

    $\mu$ and $\Sigma$ are partitioned as follows

    $$
    \mu =
    \begin{bmatrix}
    \mu_{acc} \\
    \mu_{rel}
    \end{bmatrix}
    \quad \text{with sizes} \quad
    \begin{bmatrix}
    3 \times 1 \\
    2 \times 1
    \end{bmatrix}
    $$

    $$
    \Sigma =
    \begin{bmatrix}
    \Sigma_{acc} & \Sigma_{cross} \\
    \Sigma_{cross}^T & \Sigma_{rel}
    \end{bmatrix}
    \quad \text{with sizes} \quad
    \begin{bmatrix}
    3 \times 3 & 3 \times 2 \\
    2 \times 3 & 2 \times 2
    \end{bmatrix}
    $$


   $\Sigma_{cross}$ is the cross covariance matrix between $\Sigma_{acc}$ and $\Sigma_{rel}$.
         

         
    $\mathbf{x}_{acc}$ conditional on observed release position $a$ is multivariate normal
         
    $$
    \mathbf{x}_{acc} \mid \mathbf{x}_{rel} = a \sim \mathcal{N}(\bar{\mu}, \bar{\Sigma})
    $$

    where

    $$
    \bar{\mu} = \mu_{rel} + \Sigma_{cross}\Sigma_{acc}^{-1}(a - \mu_{acc})
    $$

    and covariance matrix

    $$
    \bar{\Sigma} = \Sigma_{rel} - \Sigma_{cross}\Sigma_{acc}^{-1}\Sigma_{cross}^T
    $$
    '''

)

