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

def plot_confidence_ellipse(mu, cov, alph, ax, clabel=None, label_bg='white', clabel_size=10, **kwargs):
    c = -2 * np.log(1 - alph)
    Lambda, Q = la.eig(cov)
    width, height = 2 * np.sqrt(c * Lambda)
    theta = 180 * np.arctan(Q[1,0] / Q[0,0]) / np.pi if cov[1,0] else 0
    if 'fc' not in kwargs.keys():
        kwargs['fc'] = 'None'
    level_line = Ellipse(mu, width, height, angle=theta, **kwargs)
    if clabel:
        col = kwargs['ec'] if 'ec' in kwargs.keys() and kwargs['ec'] != 'None' else 'black'
        pos = Q[:,1] * np.sqrt(c * Lambda[1]) + mu
        ax.text(*pos, clabel, color=col, rotation=theta, ha='center', va='center', rotation_mode='anchor',
                fontsize=clabel_size, bbox=dict(boxstyle='round',ec='None',fc=label_bg, alpha=1))
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

# Streamlit app
#st.sidebar.title("Page")

page = st.sidebar.radio("Choose one of", ["DDZ Visualization", "Explainer"])

if page == "DDZ Visualization":

    st.title("Dynamic Dead Zone Visualization")
    st.write("This app visualizes the shape(s) of a pitcher's fastball(s) relative to the expected shape given the pitcher's release position. You can select a pitcher and visualize the shape of their fastball for different years. A more unlikely shape is likely to be more surprising to the batter and more effective at limiting damage.")
    st.text('')
    st.write('I use release-direction-relative acceleration as the operant metric for shape because it is independent of time under a constant acceleration model, and I did not want time to plate as a source of variance. You can think of the acceleration components as, basically, induced vertical break and horizontal break.')
    st.markdown('####')

    pitcher_names = sorted(considered_players['player_name'].unique().to_numpy())
    pitcher_name = st.selectbox("Select a player", pitcher_names)
    years = sorted(considered_players.filter(pl.col('player_name') == pitcher_name)['game_year'].unique().to_numpy())
    year = st.selectbox("Select a year", years)

    if st.button("Generate Visualization"):
        pitcher = considered_players.filter(pl.col('player_name') == pitcher_name)['pitcher'].to_numpy()[0]
        
        mu_expected_ff = player_params[pitcher][year]['FF']['mu_expected']
        sig_expected_ff = player_params[pitcher][year]['FF']['sig_expected']
        mu_expected_si = player_params[pitcher][year]['SI']['mu_expected']
        sig_expected_si = player_params[pitcher][year]['SI']['sig_expected']
        mu_expected_fc = player_params[pitcher][year]['FC']['mu_expected']
        sig_expected_fc = player_params[pitcher][year]['FC']['sig_expected']

        arm_angle = considered_players.filter((pl.col('player_name') == pitcher_name) & (pl.col('game_year') == year))['arm_angle'].to_numpy()[0]
        rel_vals = considered_players.filter((pl.col('player_name') == pitcher_name) & (pl.col('game_year') == year))[['arm_angle','extension_ratio']].to_numpy()[0]
        pitcher_name = reverse_name(pitcher_name)

        fig, ax = plt.subplots(figsize=(18, 12))
        x = np.linspace(-15, 25, 100)
        y = np.linspace(-5, 25, 100)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))

        p_fc, p_si, p_ff = tuple(model_lr.predict_proba(rel_vals[None,:])[0])

        rv_ff = multivariate_normal(mu_expected_ff, sig_expected_ff)
        Z_ff = rv_ff.pdf(pos) * p_ff
        rv_si = multivariate_normal(mu_expected_si, sig_expected_si)
        Z_si = rv_si.pdf(pos) * p_si
        rv_fc = multivariate_normal(mu_expected_fc, sig_expected_fc)
        Z_fc = rv_fc.pdf(pos) * p_fc
        Z = Z_ff + Z_si + Z_fc

        n_levels = 6
        contour_color_lst = ['#000000','#1E3F5A', '#265D8C', '#3182BD', '#6BAED6', '#9ECAE1']
        custom_cmap = colors.ListedColormap(contour_color_lst)
        levels = np.linspace(Z.min(), Z.max(), n_levels + 1)
        contours = ax.contourf(X, Y, Z, levels=levels, cmap=custom_cmap, extend='both')

        example_mu = np.array([20,28])
        example_sig = np.array([[1,0],[0,1]])*3
        plot_confidence_ellipse(example_mu, example_sig, 0.65, ax, ec='white', label_bg=contour_color_lst[0], clabel='Pitch Type', clabel_size=10)
        ax.text(*example_mu, 'Unexpected\nRide/Run', color='white', ha='center', va='center', fontsize=9)

        pt_colors = ['#ff828b','#e7c582','#00b0ba']
        counted_types = 0
        for i, pitch_type in enumerate(['FF','SI','FC']):
            if len(player_params[pitcher][year][pitch_type]) > 2:
                counted_types += 1
                mu_actual = player_params[pitcher][year][pitch_type]['mu_actual']
                sig_actual = player_params[pitcher][year][pitch_type]['sig_actual']
                plot_confidence_ellipse(mu_actual, sig_actual, .7, ax, ec=pt_colors[i], label_bg=contour_color_lst[0], clabel=pitch_type, clabel_size=14)
                
                if pitch_type == 'FF':
                    ride_run_resid = np.round((mu_actual - mu_expected_ff),1)
                elif pitch_type == 'SI':
                    ride_run_resid = np.round((mu_actual - mu_expected_si),1)
                elif pitch_type == 'FC':
                    ride_run_resid = np.round((mu_actual - mu_expected_fc),1)
                
                ride_run_resid_text = f'{"+" if ride_run_resid[1] > 0 else "-"}{np.abs(ride_run_resid[1])}/{"+" if ride_run_resid[0] > 0 else "-"}{np.abs(ride_run_resid[0])}'
                ax.text(mu_actual[0], mu_actual[1], ride_run_resid_text, color=pt_colors[i], ha='center', va='center', fontweight='bold', fontsize=10)

        ax.set_xlim(-10, 25)
        ax.set_ylim(-5, 30)
        ax.set_aspect('equal')
        ax.axhline(0, color='white', linestyle='-')
        ax.axvline(0, color='white', linestyle='-')
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Horizontal acceleration (ft/s²),\nrelative to release direction", fontsize=10)
        ax.set_ylabel("Vertical acceleration (ft/s²),\nrelative to release direction", fontsize=10)
        plt.suptitle(f"{pitcher_name}'s fastball shape{'' if counted_types == 1 else 's'} relative\nto expected given arm angle and extension ({year})", 
                    fontsize=18, fontweight='bold',color = 'white',y = 1)

        cbar_ax = fig.add_axes([0.6, 0.9, 0.1, 0.02])
        cbar = fig.colorbar(contours, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks([])
        cbar_ax.text(-.1, 0.5, "Most\nexpected", transform=cbar_ax.transAxes, fontsize=10, ha='right', va='center')
        cbar_ax.text(1.1, 0.5, "Least\nexpected", transform=cbar_ax.transAxes, fontsize=10, ha='left', va='center')
        cbar_ax.invert_xaxis()
        cbar.outline.set_edgecolor('#dddddd')
        cbar.outline.set_linewidth(2)

        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')

        for text in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('white')
            text.set_fontsize(text.get_fontsize() + 2)

        cbar_ax.yaxis.label.set_color('white')
        for text in cbar_ax.texts:
            text.set_color('white')

        arm_angle_from_midnight = 90 - arm_angle
        arm_angle_rad = np.deg2rad(arm_angle_from_midnight)
        line_length = 30
        x_line = [0, line_length * np.cos(arm_angle_rad)]
        y_line = [0, line_length * np.sin(arm_angle_rad)]
        ax.plot(x_line, y_line, linestyle='--', color='#dddddd')

        mid_x = line_length * np.cos(arm_angle_rad) * .2
        mid_y = line_length * np.sin(arm_angle_rad)  * .2
        text_offset = 1
        ax.text(mid_x, mid_y + text_offset, f'Arm Angle: {round(arm_angle,1)}°', color='#dddddd', fontsize=12, rotation=arm_angle_from_midnight, rotation_mode='anchor', ha='center', va='center')

        ax.text(1.2, -.2, "Model by Max Bay\nData from baseballsavant.mlb.com", transform=ax.transAxes, fontsize=10, ha='right', va='bottom', color='white')

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        bar_ax = fig.add_axes([.31, 0.885, 0.07, 0.05])
        pitch_types = ['FF', 'SI', 'FC']
        probs = [p_ff, p_si, p_fc]
        bar_ax.bar(pitch_types, probs, color=pt_colors)

        bar_ax.set_facecolor('black')
        bar_ax.tick_params(axis='x', colors='white')
        bar_ax.tick_params(axis='y', colors='white')
        bar_ax.set_ylim(0, 1)
        for spine in bar_ax.spines.values():
            spine.set_color('white')

        bar_ax.set_title('Expected Pitch Type', color='white', fontsize=9)

        st.pyplot(fig)

elif page == "Explainer":
    st.title("Explainer: A Mixture of Conditional Multivariate Normal Distributions")

    st.write(r'''
    For any given pitch type, we model the release characteristics arm angle $a$, pitcher height-scaled extension $\hat{e} = \frac{e}{h}$, and pitch acceleration $(a_x,a_z)$ jointly as a 4-dimensional multivariate normal distribution.
    Let $\mathbf{X}_{\text{pitch type}}$ represent this joint distribution:

    $$
    \mathbf{X}_{\text{pitch type}} \sim \mathcal{N}(\mu, \Sigma)
    $$

    We partition $\mathbf{X}$ into release characteristics and acceleration components:

    $$
    \mathbf{X} =
    \begin{bmatrix}
    \mathbf{X}_{acc} \\
    \mathbf{X}_{rel}
    \end{bmatrix}
    $$

    The mean $\mu$ and covariance matrix $\Sigma$ are partitioned accordingly:

    $$
    \mu =
    \begin{bmatrix}
    \mu_{acc} \\
    \mu_{rel}
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
    \Sigma_{acc} & \Sigma_{cross} \\
    \Sigma_{cross}^T & \Sigma_{rel}
    \end{bmatrix}
    \quad \text{with sizes} \quad
    \begin{bmatrix}
    2 \times 2 & 2 \times 2 \\
    2 \times 2 & 2 \times 2
    \end{bmatrix}
    $$

    Here, $\Sigma_{cross}$ is the cross-covariance matrix between $\mathbf{X}_{acc}$ and $\mathbf{X}_{rel}$.

    Given observed release characteristics $\mathbf{r} = (a, e)$, the conditional distribution of acceleration $\mathbf{X}_{acc}$ is:

    $$
    \mathbf{X}_{acc} \mid \mathbf{X}_{rel} = \mathbf{r} \sim \mathcal{N}(\bar{\mu}, \bar{\Sigma})
    $$

    where:

    $$
    \bar{\mu} = \mu_{acc} + \Sigma_{cross}\Sigma_{rel}^{-1}(\mathbf{r} - \mu_{rel})
    $$

    $$
    \bar{\Sigma} = \Sigma_{acc} - \Sigma_{cross}\Sigma_{rel}^{-1}\Sigma_{cross}^T
    $$

    We perform this conditioning separately for four-seam fastballs (FF), sinkers (SI), and cutters (FC) to produce $\mathbf{X}_{FF}$, $\mathbf{X}_{SI}$, and $\mathbf{X}_{FC}$.

    The final model is a mixture of these conditional distributions:

    $$
    \mathbf{X} \sim \pi_{FF} \mathcal{N}(\bar{\mu}_{FF}, \bar{\Sigma}_{FF}) + \pi_{SI} \mathcal{N}(\bar{\mu}_{SI}, \bar{\Sigma}_{SI}) + \pi_{FC} \mathcal{N}(\bar{\mu}_{FC}, \bar{\Sigma}_{FC})
    $$

    The mixing weights $\pi_{FF}$, $\pi_{SI}$, and $\pi_{FC}$ represent the probability of each pitch type given the release characteristics. We calculate these weights using multinomial logistic regression:

    $$
    \pi_k = P(Z = k \mid \mathbf{r}) = \frac{e^{\mathbf{r}^T \beta_k}}{\sum_{j \in \{FF, SI, FC\}} e^{\mathbf{r}^T \beta_j}}
    $$

    where:
    - $Z$ is the categorical variable representing pitch type
    - $k \in \{FF, SI, FC\}$
    - $\mathbf{r} = (a, e)$ is the input vector of release characteristics (arm angle and scaled extension)
    - $\beta_k$ are the learned coefficients for each pitch type
    ''')