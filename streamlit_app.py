import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import matplotlib.colors as colors
import pickle
import joblib
import polars as pl
from numpy import linalg as la
from matplotlib.patches import Ellipse
from matplotlib.ticker import FuncFormatter
from PIL import Image


# Helper functions
def reverse_name(name):
    parts = name.split(', ')
    if len(parts) == 1:
        return name
    last = parts[0]
    first_and_suffix = parts[1].split()
    if len(first_and_suffix) > 1 and first_and_suffix[-1] in ['Jr.', 'Sr.', 'II', 'III', 'IV']:
        first = ' '.join(first_and_suffix[:-1])
        suffix = first_and_suffix[-1]
        return f"{first} {last} {suffix}"
    else:
        first = ' '.join(first_and_suffix)
        return f"{first} {last}"

def plot_confidence_ellipse(mu, cov, alph, ax, clabel=None, clabel_opposite=None, label_bg='white', clabel_size=10, **kwargs):
    """Display a confidence ellipse of a bivariate normal distribution
    
    Arguments:
        mu {array-like of shape (2,)} -- mean of the distribution
        cov {array-like of shape(2,2)} -- covariance matrix
        alph {float btw 0 and 1} -- level of confidence
        ax {plt.Axes} -- axes on which to display the ellipse
        clabel {str} -- label to add to ellipse (default: {None})
        clabel_opposite {str} -- label to add to the opposite side of the ellipse (default: {None})
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
    
    offset = 0.25
    ## Display a label 'clabel' on the ellipse
    if clabel:
        col = kwargs['ec'] if 'ec' in kwargs.keys() and kwargs['ec'] != 'None' else 'black'  # color of the text
          # Adjust this value to push the text further out
        pos = Q[:,1] * (np.sqrt(c * Lambda[1]) + offset) + mu  # position along the heigth
        
        ax.text(*pos, '                  ', color=col,
           rotation=theta, ha='center', va='center', rotation_mode='anchor', # rotation
           fontsize=clabel_size,  # set the font size
           bbox=dict(boxstyle='round',ec='None',fc=label_bg, alpha=.25)) # white box
        
        ax.text(*pos, clabel, color=col,
           rotation=theta, ha='center', va='center', rotation_mode='anchor', # rotation
           fontsize=clabel_size) 

    ## Display a label 'clabel_opposite' on the opposite side of the ellipse
    if clabel_opposite:
        col = kwargs['ec'] if 'ec' in kwargs.keys() and kwargs['ec'] != 'None' else 'black'  # color of the text
        pos_opposite = -Q[:,1] * (np.sqrt(c * Lambda[1]) + offset) + mu  # position on the opposite side
        
        ax.text(*pos_opposite, '                  ', color=col,
           rotation=theta, ha='center', va='center', rotation_mode='anchor', # rotation
           fontsize=clabel_size,  # set the font size
           bbox=dict(boxstyle='round',ec='None',fc=label_bg, alpha=.25)) # white box
        
        ax.text(*pos_opposite, clabel_opposite, color=col,
           rotation=theta, ha='center', va='center', rotation_mode='anchor', # rotation
           fontsize=clabel_size) 
        
    return ax.add_patch(level_line)

# Load data
@st.cache_data
def load_data():
    considered_players = pl.read_csv('./data/considered_players.csv')
    with open('./data/params.pkl', 'rb') as f:
        player_params = pickle.load(f)
    model_lr = joblib.load('./data/model_lr.pkl')
    return considered_players, player_params, model_lr

considered_players, player_params, model_lr = load_data()

st.title("Dynamic Dead Zone Visualization")
st.write("This app visualizes the shape(s) of a pitcher's fastball(s) relative to the expected shape given the pitcher's release position. You can select a pitcher and visualize the shape of their fastball for different years. A more unlikely shape is likely to be more surprising to the batter and more effective at limiting damage.")
st.text('')
st.write('I use release-direction-relative acceleration as the operant metric for shape because it is independent of time under a constant acceleration model, and I did not want time to plate as a source of variance. You can think of the acceleration components as, basically, induced vertical break and horizontal break.')
st.markdown('####')

pitcher_names = sorted(considered_players['player_name'].unique().to_numpy())
pitcher_name = st.selectbox("Select a player", pitcher_names)
col1, col2 = st.columns(2)

# In the first column, place the year selection
with col1:
    years = sorted(considered_players.filter(pl.col('player_name') == pitcher_name)['game_year'].unique().to_numpy())
    year = st.selectbox("Select a year", years)

# In the second column, place the radio button
with col2:
    genre = st.radio(
        "Customize expected pitch type weight | Model expectation or concentrate all into selected pitch type",
        ["Model", "FF", "SI", "FC"],
        horizontal=True,
        index=0,
    )

if st.button("Generate Visualization"):

    pitcher = considered_players.filter(pl.col('player_name') == pitcher_name)['pitcher'].to_numpy()[0]


    t = 0.4  # seconds
    k = 0.5 * t**2
    mu_rescale = k * 12
    sig_rescale = k**2 * 12**2

    mu_expected_ff = player_params[pitcher][year]['FF']['mu_expected'] * mu_rescale
    sig_expected_ff = player_params[pitcher][year]['FF']['sig_expected'] * sig_rescale

    mu_expected_si = player_params[pitcher][year]['SI']['mu_expected'] * mu_rescale
    sig_expected_si = player_params[pitcher][year]['SI']['sig_expected'] * sig_rescale

    mu_expected_fc = player_params[pitcher][year]['FC']['mu_expected'] * mu_rescale
    sig_expected_fc = player_params[pitcher][year]['FC']['sig_expected'] * sig_rescale

    #actual
    #mu_actual = player_params[pitcher][year]['mu_actual']
    #sig_actual = player_params[pitcher][year]['sig_actual']

    # Extract wasserstein distance and whiff% from considered_players
    arm_angle = considered_players.filter((pl.col('player_name') == pitcher_name) & (pl.col('game_year') == year))['arm_angle'].to_numpy()[0]

    rel_vals = considered_players.filter((pl.col('player_name') == pitcher_name) & (pl.col('game_year') == year))[['arm_angle','extension_ratio']].to_numpy()[0]

    pitcher_name = reverse_name(pitcher_name)
    #pitcher_name = 'Walker Buehler'


    # Set up the plot
    fig, ax = plt.subplots(figsize=(18, 12))  # Increased figure height

    # Define grid for contour plot
    x = np.linspace(-15, 25, 100)
    y = np.linspace(-10, 25, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))




    if genre == "Model":
        p_fc, p_si, p_ff = tuple(model_lr.predict_proba(rel_vals[None,:])[0])
    elif genre == "FF":
        p_fc, p_si, p_ff = 0, 0, 1
    elif genre == "SI":
        p_fc, p_si, p_ff = 0, 1, 0
    elif genre == "FC":
        p_fc, p_si, p_ff = 1, 0, 0

    # Compute the bivariate normal distribution
    rv_ff = multivariate_normal(mu_expected_ff, sig_expected_ff)
    Z_ff = rv_ff.pdf(pos) * p_ff

    rv_si = multivariate_normal(mu_expected_si, sig_expected_si)
    Z_si = rv_si.pdf(pos) * p_si

    rv_fc = multivariate_normal(mu_expected_fc, sig_expected_fc)
    Z_fc = rv_fc.pdf(pos) * p_fc

    Z = Z_ff + Z_si + Z_fc

    # Create a custom colormap that matches the contour colors
    n_levels = 6
    contour_color_lst = ['#000000','#1E3F5A', '#265D8C', '#3182BD', '#6BAED6', '#9ECAE1']
    custom_cmap = colors.ListedColormap(contour_color_lst)

    # Plot the contours with the custom color scheme
    levels = np.linspace(Z.min(), Z.max(), n_levels + 1)
    contours = ax.contourf(X, Y, Z, levels=levels, cmap=custom_cmap, extend='both')

    example_mu = np.array([20.8,27.5])
    example_sig = np.array([[1,0],[0,1]])*2.5

    ellipse = plot_confidence_ellipse(example_mu, example_sig, 0.65, ax, ec='white', label_bg=contour_color_lst[0],clabel_opposite='Unexpected\nRide/Run', clabel_size=10)
    ax.text(*example_mu, 'Pitch\nType', 
            color='white', 
            ha='center', va='center', fontsize=12)

    pt_colors = ['#ff828b','#e7c582','#00b0ba']
    counted_types = 0
    for i, pitch_type in enumerate(['FF','SI','FC']):
        if len(player_params[pitcher][year][pitch_type]) > 2:
            counted_types += 1
            mu_actual = player_params[pitcher][year][pitch_type]['mu_actual'] * mu_rescale
            sig_actual = player_params[pitcher][year][pitch_type]['sig_actual'] * sig_rescale


            if pitch_type == 'FF':
                ride_run_resid = np.round((mu_actual - mu_expected_ff),1)
                mu_expected = mu_expected_ff
            elif pitch_type == 'SI':
                ride_run_resid = np.round((mu_actual - mu_expected_si),1)
                mu_expected = mu_expected_si
            elif pitch_type == 'FC':
                ride_run_resid = np.round((mu_actual - mu_expected_fc),1)
                mu_expected = mu_expected_fc

            ax.plot([mu_expected[0], mu_actual[0]], [mu_expected[1], mu_actual[1]], color=pt_colors[i], linestyle='-', linewidth=1)

            ride_run_resid_text = f'{"+" if ride_run_resid[1] > 0 else "-"}{np.abs(ride_run_resid[1])}"/{"+" if ride_run_resid[0] > 0 else "-"}{np.abs(ride_run_resid[0])}"'

            ax.scatter(*mu_expected, color=pt_colors[i], s=25)
            
            # Add centered text at the scattered point
            
            ax.text(*mu_actual, '   ', 
                    color=pt_colors[i], 
                    ha='center', va='center', 
                    fontweight='bold', fontsize=14,
                    bbox=dict(boxstyle='round',ec='None',fc=contour_color_lst[0], alpha=.25)
                    )
            
            ax.text(*mu_actual, pitch_type, 
                    color=pt_colors[i], 
                    ha='center', va='center', 
                    fontweight='bold', fontsize=14
                    )
            
            clabel_size = 11
            ellipse = plot_confidence_ellipse(mu_actual, sig_actual, .7, ax, ec=pt_colors[i], label_bg=contour_color_lst[0], clabel=ride_run_resid_text, clabel_size=clabel_size)


    # Customize the plot
    ax.set_xlim(-10, 25)
    ax.set_ylim(-10, 30)
    ax.set_aspect('equal')
    ax.axhline(0, color='white', linestyle='-')
    ax.axvline(0, color='white', linestyle='-')
    # Add gridlines
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    def inches_formatter(x, pos):
        return f'{int(x)}"'

    # Apply the formatter to both axes
    ax.xaxis.set_major_formatter(FuncFormatter(inches_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(inches_formatter))

    ax.set_xlabel("Horizontal Break ($in$)", fontsize=10)
    ax.set_ylabel("Induced Vertical Break ($in$)", fontsize=10)

    plt.suptitle(f"{pitcher_name}'s fastball shape{'' if counted_types == 1 else 's'} relative\nto expected given arm angle ({year})", 
                fontsize=18, fontweight='bold',color = 'white',y = 1)  # Increased pad to move title up


    # Create a new axes for the colorbar
    cbar_ax = fig.add_axes([0.6, 0.9, 0.1, 0.02])  # [left, bottom, width, height]

    # Add a colorbar
    cbar = fig.colorbar(contours, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([])  # Remove default ticks

    # Add color bar labels
    cbar_ax.text(-.1, 0.5, "Most\nexpected", transform=cbar_ax.transAxes, fontsize=10, ha='right', va='center')
    cbar_ax.text(1.1, 0.5, "Least\nexpected", transform=cbar_ax.transAxes, fontsize=10, ha='left', va='center')

    # Rotate the colorbar
    cbar_ax.invert_xaxis()
    cbar.outline.set_edgecolor('#dddddd')
    cbar.outline.set_linewidth(2)

    # Set background color to black
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Change text color to white and increase font size
    for text in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
        text.set_color('white')
        text.set_fontsize(text.get_fontsize() + 2)


    # Change colorbar text color to white
    cbar_ax.yaxis.label.set_color('white')
    for text in cbar_ax.texts:
        text.set_color('white')

    # Add arm angle as a grey dashed line
    arm_arngle_from_midnight = 90 - arm_angle
    arm_angle_rad = np.deg2rad(arm_arngle_from_midnight )  # Convert to radians if necessary
    line_length = 30  # Length of the line
    x_line = [0, line_length * np.cos(arm_angle_rad)]
    y_line = [0, line_length * np.sin(arm_angle_rad)]
    ax.plot(x_line, y_line, linestyle='--', color='#dddddd')

    mid_x = line_length * np.cos(arm_angle_rad) * .2
    mid_y = line_length * np.sin(arm_angle_rad)  * .2
    text_offset = 1  # Offset to place the text just above the line
    ax.text(mid_x, mid_y + text_offset, f'Arm Angle: {round(arm_angle,1)}°', color='#dddddd', fontsize=12, rotation=arm_arngle_from_midnight, rotation_mode='anchor', ha='center', va='center')

    # subtitle
    ax.text(1, -.1, "Model by Max Bay\nData from baseballsavant.mlb.com", transform=ax.transAxes, fontsize=10, ha='right', va='bottom', color='white')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjusted rect to make room at top and bottom

    bar_ax = fig.add_axes([.31, 0.885, 0.07, 0.05])  # [left, bottom, width, height] for the bar plot position
    pitch_types = ['FF', 'SI', 'FC']
    probs = [p_ff, p_si, p_fc]
    colors = pt_colors  # Use your defined pitch type colors

    # Create a horizontal bar plot
    bar_ax.bar(pitch_types, probs, color=colors)

    # Customize the embedded plot
    bar_ax.set_facecolor('black')  # Match the background color of the main plot
    bar_ax.tick_params(axis='x', colors='white')  # White ticks
    bar_ax.tick_params(axis='y', colors='white')  # White ticks
    bar_ax.set_ylim(0, 1)  # Probability should be between 0 and 1
    bar_ax.spines['top'].set_color('white')
    bar_ax.spines['right'].set_color('white')
    bar_ax.spines['bottom'].set_color('white')
    bar_ax.spines['left'].set_color('white')

    # Set bar labels
    bar_ax.set_title('Expected Pitch Type', color='white', fontsize=9)

    plt.show()

    st.pyplot(fig)

st.markdown('---')

st.markdown('<h2 style="font-size: 24px;">Theory of the case</h2>', unsafe_allow_html=True)
with st.expander("Pull down to expand"):

    st.write('''
    Empirically, break profiles are correlated with arm angle at release. This model assumes that batters have some level of awareness of this relationship, and that arm angle information primes batters to expect certain break profiles.
    ''')    
    st.image("./imgs/pitch_animation_slow.gif", caption='Arm angle vs. break')



st.markdown('<h2 style="font-size: 24px;">Mathematical explainer</h2>', unsafe_allow_html=True)
with st.expander("Pull down to expand"):
    st.write(r'''
    For any given pitch type, we model the release characteristics—arm angle $d$ and pitcher height-scaled extension $\hat{e} = \frac{e}{h}$—along with pitch acceleration components $(a_x, a_z)$ jointly as a 4-dimensional multivariate normal distribution. Let $\mathbf{X}_{\text{pitch type}}$ represent this joint distribution:

    $$
    \mathbf{X}_{\text{pitch type}} \sim \mathcal{N}(\mu, \Sigma)
    $$

    We partition $\mathbf{X}$ into acceleration and release characteristics components:

    $$
    \mathbf{X} =
    \begin{bmatrix}
    \mathbf{X}_{\text{acc}} \\
    \mathbf{X}_{\text{rel}}
    \end{bmatrix}
    $$

    where $\mathbf{X}_{\text{acc}} = \begin{bmatrix} a_x \\ a_z \end{bmatrix}$ and $\mathbf{X}_{\text{rel}} = \begin{bmatrix} d \\ \hat{e} \end{bmatrix}$.

    The mean vector $\mu$ and covariance matrix $\Sigma$ are partitioned accordingly:

    $$
    \mu =
    \begin{bmatrix}
    \mu_{\text{acc}} \\
    \mu_{\text{rel}}
    \end{bmatrix}
    \quad \text{with sizes} \quad
    \begin{bmatrix}
    2 \times 1 \\
    2 \times 1
    \end{bmatrix}
    $$

    $$
    \Sigma =
    \begin{bmatrix}
    \Sigma_{\text{acc}} & \Sigma_{\text{cross}} \\
    \Sigma_{\text{cross}}^\top & \Sigma_{\text{rel}}
    \end{bmatrix}
    \quad \text{with sizes} \quad
    \begin{bmatrix}
    2 \times 2 & 2 \times 2 \\
    2 \times 2 & 2 \times 2
    \end{bmatrix}
    $$

    Here, $\Sigma_{\text{cross}}$ is the cross-covariance matrix between $\mathbf{X}_{\text{acc}}$ and $\mathbf{X}_{\text{rel}}$.

    Given observed release characteristics $\mathbf{r} = \begin{bmatrix} d \\ \hat{e} \end{bmatrix}$, the conditional distribution of acceleration $\mathbf{X}_{\text{acc}}$ is:

    $$
    \mathbf{X}_{\text{acc}} \mid \mathbf{X}_{\text{rel}} = \mathbf{r} \sim \mathcal{N}(\bar{\mu}, \bar{\Sigma})
    $$

    where:

    $$
    \bar{\mu} = \mu_{\text{acc}} + \Sigma_{\text{cross}} \Sigma_{\text{rel}}^{-1} (\mathbf{r} - \mu_{\text{rel}})
    $$

    $$
    \bar{\Sigma} = \Sigma_{\text{acc}} - \Sigma_{\text{cross}} \Sigma_{\text{rel}}^{-1} \Sigma_{\text{cross}}^\top
    $$

    We perform this conditioning separately for four-seam fastballs (FF), sinkers (SI), and cutters (FC) to obtain $\mathbf{X}_{\text{FF}}$, $\mathbf{X}_{\text{SI}}$, and $\mathbf{X}_{\text{FC}}$.

    The final model is a mixture of these conditional distributions:

    $$
    P(\mathbf{X}_{\text{acc}} \mid \mathbf{r}) = \pi_{\text{FF}} \mathcal{N}(\bar{\mu}_{\text{FF}}, \bar{\Sigma}_{\text{FF}}) + \pi_{\text{SI}} \mathcal{N}(\bar{\mu}_{\text{SI}}, \bar{\Sigma}_{\text{SI}}) + \pi_{\text{FC}} \mathcal{N}(\bar{\mu}_{\text{FC}}, \bar{\Sigma}_{\text{FC}})
    $$

    The mixing weights $\pi_{\text{FF}}$, $\pi_{\text{SI}}$, and $\pi_{\text{FC}}$ represent the probabilities of each pitch type given the release characteristics. We calculate these weights using multinomial logistic regression:

    $$
    \pi_k = P(Z = k \mid \mathbf{r}) = \frac{e^{\mathbf{r}^\top \beta_k}}{\sum_{j \in \{\text{FF}, \text{SI}, \text{FC}\}} e^{\mathbf{r}^\top \beta_j}}
    $$

    where:
    - $Z$ is the categorical variable representing pitch type
    - $k \in \{\text{FF}, \text{SI}, \text{FC}\}$
    - $\mathbf{r} = \begin{bmatrix} d \\ \hat{e} \end{bmatrix}$ is the input vector of release characteristics (arm angle and scaled extension)
    - $\beta_k$ are the learned coefficient vectors for each pitch type
    ''')

