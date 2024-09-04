import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle
import joblib
import polars as pl
from streamlit_searchbox import st_searchbox
from scipy.stats import multivariate_normal
from sklearn.neighbors import KNeighborsClassifier
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

st.title("Expected vs Observed FF Shape")

st.write("This app visualizes the shape of a pitcher's four-seam fastball (FF) relative to the expected shape given the picher's release position. You can select a pitcher and visualize the shape of their fastball for different years. A more unlikely shape is likely to be more surprising to the batter and more effective at limiting damage.")
st.text('')
st.write('I use release-direction-relative acceleration as the operant metric for shape because it is independent of time under a constant acceleration model, and I did not want time to plate as a source of variance. You can think of the acceleration components as, basically, induced vertical break and horizontal break.')
st.markdown('####')
# get player params
dict_path = './data/params.pkl'
with open(dict_path, 'rb') as f:
    player_params = pickle.load(f)

dict_path_si = './data/params_si.pkl'
with open(dict_path_si, 'rb') as f:
    player_params_si = pickle.load(f)

model_knn = joblib.load('./data/p_ff_mdl.pkl')


# get player df
df_path = './data/considered_players.csv'
considered_playeryears = pl.read_csv(df_path)

# player Selection with Autocomplete
players = considered_playeryears['player_name'].unique().to_list()

selected_player = st_searchbox(search_players, label="Select a Pitcher",default='Cole, Gerrit')

years = considered_playeryears.filter(pl.col('player_name') == selected_player)['game_year'].unique().to_list()

st.write(f"Select the years you want to visualize:")

cols = st.columns(len(years))

checkboxes = {}
for i, year in enumerate(years):
    checkboxes[year] = cols[i].checkbox(str(year), value=(year == max(years)))

selected_years = [year for year in years if checkboxes[year]]


if any([checkboxes[year] for year in years]):

    f, ax = plt.subplots(1, 1, figsize=(10, 10))

    latest_year = max(selected_years)

    colors = get_spectral_colors(5, pal = 'Set1')
    #######

    pitcher_name = selected_player
    pitcher = considered_playeryears.filter(pl.col('player_name') == pitcher_name)['pitcher'].to_numpy()[0]

    rel_vals = considered_playeryears.filter((pl.col('player_name') == pitcher_name) & (pl.col('game_year') == latest_year))[['release_pos_x_adj','release_extension_adj','release_pos_z_adj']].to_numpy()[0]

    # Parameters
    #expected
    mu_expected = player_params[pitcher][latest_year]['mu_expected']
    sig_expected = player_params[pitcher][latest_year]['sig_expected']

    mu_expected_si = player_params_si[pitcher][latest_year]['mu_expected']
    sig_expected_si = player_params_si[pitcher][latest_year]['sig_expected']

    
    clabel_size = 7

    # Define grid for contour plot
    x = np.linspace(-5, 20, 100)
    y = np.linspace(-5, 25, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    p_ff = model_knn.predict_proba(rel_vals[None,:])[:,1]

    # compute the bivariate normal distribution
    rv_ff = multivariate_normal(mu_expected, sig_expected)
    Z_ff = rv_ff.pdf(pos) * p_ff

    rv_si = multivariate_normal(mu_expected_si, sig_expected_si)
    Z_si = rv_si.pdf(pos) * (1 - p_ff)

    Z = Z_ff + Z_si

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
st.write("### Explainer... it's just a mixture of conditional MVNs")

st.write('')
st.write(r'''


    For any given pitch type, the pitcher-height-scaled pitch type population release position $(x',y',z') = \frac{(x,y,z)}{height}$ and pitch acceleration $(a_x,a_z)$ can be jointly modeled as a $5$-dimensional multivariate normal distribution $X_{\text{pitch type}}$ .

    $$
    X_{\text{pitch type}} \sim  \mathcal{N}(\mu, \Sigma)
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
    2 \times 1 \\
    3 \times 1
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
    2 \times 2 & 2 \times 3 \\
    3 \times 2 & 3 \times 3
    \end{bmatrix}
    $$


   $\Sigma_{cross}$ is the cross covariance matrix between $\mathbf{x}_{acc}$ and $\mathbf{x}_{rel}$.
         

         
    $\mathbf{x}_{acc}$ conditional on observed release position $a$ is multivariate normal
         
    $$
    \mathbf{x}_{acc} \mid \mathbf{x}_{rel} = a \sim \mathcal{N}(\bar{\mu}, \bar{\Sigma})
    $$

    where

    $$
    \bar{\mu} = \mu_{acc} + \Sigma_{cross}\Sigma_{rel}^{-1}(a - \mu_{rel})
    $$

    and covariance matrix

    $$
    \bar{\Sigma} = \Sigma_{acc} - \Sigma_{cross}\Sigma_{rel}^{-1}\Sigma_{cross}^T
    $$
         
    The above is performed for both four-seam fastballs (FF) and sinkers (SI) independently to produce $X_{FF}$ and $X_{SI}$. Finally, the two distributions are mixed, giving the random variable $X$, where
                  
    $$
    X \sim \pi_{\text{FF}} \mathcal{N}(\bar{\mu}_{\text{FF}}, \bar{\Sigma}_{\text{FF}}) + \pi_{\text{SI}} \mathcal{N}(\bar{\mu}_{\text{SI}}, \bar{\Sigma}_{\text{SI}})
    $$

    $\pi_{FF}$ and $\pi_{SI}$ are component weights, equivalent to the probability of the pitch being a FF or SI respectively. The weights are calculated from a logistic regression fit
    
    $$
    \pi_{FF} = p(\text{FF}) = \frac{1}{1 + \exp{-(\beta_0 + \beta_1 \cdot x' + \beta_2 \cdot y' + \beta_3 \cdot z' + \beta_4)}}
    $$
    and 
         
    $$
    \pi_{SI} = 1 - \pi_{FF}
    $$
    

 
         
    '''

)

