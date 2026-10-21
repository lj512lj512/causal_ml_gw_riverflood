# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 08:53:39 2026

@author: juliu
"""
import os 
import pandas as pd
import numpy as np
import geopandas as gpd 
from A2_causal_model_train import load_dataset, make_model_y, make_model_t
from econml.dml import CausalForestDML, NonParamDML
from lightgbm import LGBMRegressor, LGBMClassifier
from shapely import force_2d
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import ee
import geemap
ee.Authenticate()
ee.Initialize()

def get_ee_features(catch):
    
    # data of year alpha earth based on
    year = 2020

    # Reproject to WGS84
    catchments = catch.to_crs(epsg=4326)
    
    # Drop Z dimension
    catch["geometry"] = catch.geometry.apply(force_2d)
    
    # to ee 
    catchments_ee = geemap.geopandas_to_ee(catch)
    
    # Load AlphaEarth / satellite embedding annual image
    alphaearth = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterDate(f"2020-01-01", f"2020-12-31")
        .filterBounds(catchments_ee.geometry())
        .mosaic() # combine all matching images into one image
    )
    
    # Extract catchment mean embeddings
    stats = alphaearth.reduceRegions(
        collection=catchments_ee,
        reducer=ee.Reducer.mean(),
        scale=10
    )
    
    # data frame 
    df_alpha = geemap.ee_to_df(stats, remove_geom=False)
    
    return df_alpha

ealpha_e_features =['A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07',
       'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17',
       'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27',
       'A28', 'A29', 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37',
       'A38', 'A39', 'A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47',
       'A48', 'A49', 'A50', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57',
       'A58', 'A59', 'A60', 'A61', 'A62', 'A63',]


#%% step 0 prepare predictors
datapath = (
    r"\\geodata.geus.dk\DKmodel_users\FloodWarning\LSTM_postprocessing"
    r"\CAMELS-DK\Level_3\Dynamics\Ungauged_catchments")

camels_3330 = gpd.read_file(r"\\geodata.geus.dk\dkmodel_users\FloodWarning\LSTM_postprocessing\CAMELS-DK\Level_3\Shapefile\CAMELS_DK_3330_catchment_boundaries.shp")

all_rows = []
for i, catch_id in enumerate(camels_3330['catch_id']):
    print(i, catch_id)
    
    ts_path = os.path.join(datapath, f"CAMELS_DK_sim_based_{catch_id}.csv")
    if not os.path.exists(ts_path):
        print(f"Missing file, skipping catchment {catch_id}: {ts_path}")
        continue

    dtp  = pd.read_csv(ts_path, index_col="time",parse_dates=True)
    dtp_r = dtp['DKM_dtp'] - dtp['DKM_dtp'].mean()
    
    # -----------------------------
    # AlphaEarth / spatial features
    # -----------------------------
    catch_geom = camels_3330.loc[camels_3330["catch_id"] == catch_id]
    alpha_features = get_ee_features(catch=catch_geom)
    
    # If get_ee_features returns a DataFrame with one row
    if isinstance(alpha_features, pd.DataFrame):
        alpha_dict = alpha_features.iloc[0].to_dict()
    
    # If it already returns a dictionary
    elif isinstance(alpha_features, dict):
        alpha_dict = alpha_features
    
    else:
        raise TypeError(
            f"Unexpected alpha_features type for catch_id {catch_id}: "
            f"{type(alpha_features)}"
        )
    
    row = {
        "catch_id": catch_id,
        "dtp_q75_value": dtp_r.quantile(0.75),
        "dtp_q90_value": dtp_r.quantile(0.90),
        "dtp_q75_q90_delta": dtp_r.quantile(0.90) - dtp_r.quantile(0.75),
        "dtp_mean": dtp["DKM_dtp"].mean(),
        "dtp_std": dtp_r.std(),
        }
    row.update(alpha_dict)
    all_rows.append(row)
    
all_catchments_df = pd.DataFrame(all_rows)
print(all_catchments_df[["catch_id", "dtp_q75_value", "dtp_q90_value", "dtp_q75_q90_delta"]].describe())

out_csv = r"data\all_catchments_groundwater_spatial_features.csv"
all_catchments_df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

#%% Step 1: Train final causal forest on gauged events
dataset = load_dataset(scale='catch') # catch, valley, downs
dataset = dataset.dropna(subset=["flood_occurrence"]).reset_index(drop=True)

target_type = 'peak' #["peak", "occurrence", "volume", "duration"]
treatment = 'dtp'    # groundwater depth relative to mean 
confounders = [ "rain_duration_days", 
               "event_precip_mm", 
               "max_1d_precip_mm", 
               "split_from_long_event", 
               "wcr", # soil water content / soil moisture 
               "sdy", # sine day of year 
               "cdy", # cosine day of year 
               "tem", # temperature 
               "pet" # potential evapotranspiration 
               ] 
heterogeneity_features = ealpha_e_features # + ["sdy","cdy"]
extra_state_cols = ["dtp_q75_value", "dtp_q90_value"]
    
if target_type == "peak":
    dataset["peak_discharge_log"] = np.log(dataset["flood_peak"] + 1e-6)
    outcome = "peak_discharge_log"
    raw_outcome = "flood_peak"
    is_binary_outcome = False
    is_log_outcome = True
    zero_heavy = False

elif target_type == "volume":
    dataset["flood_volume_above_q95_log"] = np.log1p(dataset["flood_volume_above_q95"])
    outcome = "flood_volume_above_q95_log"
    raw_outcome = "flood_volume_above_q95"
    is_binary_outcome = False
    is_log_outcome = True
    zero_heavy = True
    print("Percentage of events with flood_volume_above_q95 > 0:",100 * (dataset[raw_outcome] > 0).mean())
            
    
model_y_fold = make_model_y(target_type='peaks', random_state=5000)
model_t_fold = make_model_t(random_state=50001)
cf = NonParamDML(
    model_y=model_y_fold,
    model_t=model_t_fold,
    model_final=LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=75275,
        n_jobs=48,
        verbose=-1
    ),
    discrete_treatment=False,
    discrete_outcome=is_binary_outcome,
    cv=3,
    random_state=752)

cf.fit(Y = dataset[outcome].values, 
       T = dataset[treatment].values, 
       W = dataset[confounders].values, 
       X = dataset[heterogeneity_features].values)
                


#%% Predict Q90 sensitivity

X_all = all_catchments_df[heterogeneity_features].values

T0 = all_catchments_df["dtp_q90_value"].values
T1 = all_catchments_df["dtp_q90_value"].values + 0.1

tau_q90 = cf.effect(X_all, T0=T0, T1=T1)
tau_q90 = np.asarray(tau_q90).ravel()

all_catchments_df["sensitivity_q90_plus10"] = (np.exp(tau_q90) - 1) * 100

catchment_polygons = gpd.read_file(os.path.join(r'\\geodata.geus.dk\DKmodel_users\FloodWarning\LSTM_postprocessing\CAMELS-DK\Level_3\Shapefile', 
                                      'CAMELS_DK_3330_catchment_boundaries.shp'))
map_df = catchment_polygons.merge(
    all_catchments_df[["catch_id", "sensitivity_q90_plus10"]],
    on="catch_id",
    how="left"
)

#%% Step 4: Map it
def _subplot_setup(ax, extent, edgecolor='k', del_title=False):
    ax.set_xlabel('') #remove labels automatically created by xarray
    ax.set_ylabel('')
    ax.axes.get_xaxis().set_visible(False) #remove axis ticks
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.set_aspect('equal') #set aspect equal
    ax.set_xlim(extent[0]) #set axis limits (after xarray as elsewise will be overridden)
    ax.set_ylim(extent[1])
    for spine in ax.spines.values(): #change axes frame color
        spine.set_edgecolor(edgecolor)
    if del_title==True: #remove titles
        ax.set_title('')

def _add_inset_subplot_BH(fig, ax, rect, facecolor='w'):
    # fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2]) #
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)   
    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height],facecolor=facecolor)
    
    subax.set_xlabel('')
    subax.set_ylabel('')
    subax.axes.get_xaxis().set_visible(False)
    subax.axes.get_yaxis().set_visible(False)
    subax.axes.set_aspect('equal')
    subax.set_xlim([850000, 900000])
    subax.set_ylim([6104000, 6150000])
    return subax




def add_annular_class_legend(
    ax,
    class_counts,
    class_colors,
    center=(0.78, 0.18),
    radius=0.11,
    width=0.035,
    start_angle=90,
    title="Vulnerability class",
    fontsize=6
):
    """
    Add an annular bar legend inside a map axis.

    class_counts: pandas Series with class labels as index and counts as values
    class_colors: dict mapping class labels to colors
    center, radius, width are in axis coordinates.
    """

    total = class_counts.sum()
    angle = start_angle

    for cls, count in class_counts.items():
        if count == 0:
            continue

        frac = count / total
        theta1 = angle
        theta2 = angle - 360 * frac

        wedge = Wedge(
            center,
            radius,
            theta2,
            theta1,
            width=width,
            transform=ax.transAxes,
            facecolor=class_colors[cls],
            edgecolor="white",
            linewidth=0.4,
            zorder=20
        )

        ax.add_patch(wedge)

        angle = theta2

    # center text
    ax.text(
        center[0],
        center[1],
        f"n={int(total)}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=21
    )

    # title
    ax.text(
        center[0],
        center[1] + radius + 0.045,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        zorder=21
    )

    # class labels to the side
    y0 = center[1] - radius
    dy = 0.035

    for i, (cls, count) in enumerate(class_counts.items()):
        y = y0 + i * dy

        ax.scatter(
            center[0] + radius + 0.055,
            y,
            s=16,
            color=class_colors[cls],
            transform=ax.transAxes,
            zorder=21
        )

        ax.text(
            center[0] + radius + 0.075,
            y,
            f"{cls}: {int(count)}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=fontsize,
            zorder=21
        )
        


def add_annular_class_legend(
    ax,
    class_counts,
    class_colors,
    center=(0.78, 0.18),
    radius=0.11,
    width=0.035,
    start_angle=90,
    title="Vulnerability class",
    fontsize=6
):
    """
    Add an annular bar legend inside a map axis.

    class_counts: pandas Series with class labels as index and counts as values
    class_colors: dict mapping class labels to colors
    center, radius, width are in axis coordinates.
    """

    total = class_counts.sum()
    angle = start_angle

    for cls, count in class_counts.items():
        if count == 0:
            continue

        frac = count / total
        theta1 = angle
        theta2 = angle - 360 * frac

        wedge = Wedge(
            center,
            radius,
            theta2,
            theta1,
            width=width,
            transform=ax.transAxes,
            facecolor=class_colors[cls],
            edgecolor="white",
            linewidth=0.4,
            zorder=20
        )

        ax.add_patch(wedge)

        angle = theta2

    # center text
    ax.text(
        center[0],
        center[1],
        f"n={int(total)}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=21
    )

    # title
    ax.text(
        center[0],
        center[1] + radius + 0.045,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        zorder=21
    )

    # class labels to the side
    y0 = center[1] - radius
    dy = 0.035

    for i, (cls, count) in enumerate(class_counts.items()):
        y = y0 + i * dy

        ax.scatter(
            center[0] + radius + 0.055,
            y,
            s=16,
            color=class_colors[cls],
            transform=ax.transAxes,
            zorder=21
        )

        ax.text(
            center[0] + radius + 0.075,
            y,
            f"{cls}: {int(count)}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=fontsize,
            zorder=21
        )
    

import numpy as np
import pandas as pd
import geopandas as gpd

# --------------------------------------------------
# Create random map_gdf for testing plot design
# --------------------------------------------------

rng = np.random.default_rng(42)

map_gdf = catchment_polygons.copy()

# Make sure catch_id is string
map_gdf["catch_id"] = map_gdf["catch_id"].astype(str)

n = len(map_gdf)


# --------------------------------------------------
# Ranked vulnerability class
# --------------------------------------------------

map_gdf["vulnerability_class"] = pd.cut(
    map_gdf["combined_vulnerability_score"],
    bins=[0, 0.25, 0.50, 0.75, 0.90, 1.00],
    labels=[
        "Very low",
        "Low",
        "Moderate",
        "High",
        "Very high"
    ],
    include_lowest=True
)

# --------------------------------------------------
# Dominant vulnerability type
# --------------------------------------------------

rank_diff = map_gdf["peak_rank"] - map_gdf["volume_rank"]

conditions = [
    (map_gdf["peak_rank"] >= 0.75) & (map_gdf["volume_rank"] >= 0.75),
    rank_diff >= 0.25,
    rank_diff <= -0.25,
]

choices = [
    "High peak and volume",
    "Peak-dominated",
    "Volume-dominated",
]

map_gdf["vulnerability_type"] = np.select(
    conditions,
    choices,
    default="Mixed / moderate"
)

map_gdf.head()

class_order = ["Very low", "Low", "Moderate", "High", "Very high"]

vulnerability_colors = {
    "Very low": "#2166ac",
    "Low": "#67a9cf",
    "Moderate": "#f7f7f7",
    "High": "#f4a582",
    "Very high": "#b2182b",
}

map_gdf["vulnerability_class"] = pd.Categorical(map_gdf["vulnerability_class"],categories=class_order,ordered=True)
cmap = ListedColormap([vulnerability_colors[c] for c in class_order])
class_counts = (map_gdf["vulnerability_class"].value_counts().reindex(class_order).fillna(0))


# plt.show()
extent = {} # [[xmin, xmax], [ymin, ymax]]
extent['DK'] = [[439000, 739000], [6049000, 6404000]]
extent['BH'] = [[850000, 900000], [6104000, 6150000]]
DK  = gpd.read_file(r'\\geodata.geus.dk\HOME\Discharge_correction_with_LSTM\Shapfile\DKDomains2013\kort10_land_DK.shp')
BH  = gpd.read_file(r'\\geodata.geus.dk\HOME\Discharge_correction_with_LSTM\Shapfile\DKDomains2013\kort10_land_BH.shp')



# a. Peak-discharge sensitivity under high groundwater, Where does very high groundwater amplify flood peaks?
# b. Flood-volume sensitivity under high groundwater, Where does very high groundwater amplify flood volume?
# c. Combined vulnerability rank, Where are catchments consistently sensitive across both peak and volume?'
# d. Dominant vulnerability type, Are catchments more peak-sensitive, volume-sensitive, or both?

fig, axes = plt.subplots(1, 4,sharex=True,sharey=True,figsize=(9.5, 8.5),dpi=600,constrained_layout=False)
for ax in axes:
    _subplot_setup(ax, extent["DK"], "1.0")
    DK.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=0.35,
        zorder=5
    )
    catchment_polygons.plot(
        ax=ax,
        facecolor="none",
        edgecolor="gray",
        linewidth=0.05,
        zorder=6
    )

    
plt.tight_layout()


#%%

class_order = ["Very low", "Low", "Moderate", "High", "Very high"]

vulnerability_colors = {
    "Very low": "#2166ac",
    "Low": "#67a9cf",
    "Moderate": "#f7f7f7",
    "High": "#f4a582",
    "Very high": "#b2182b",
}

def make_rank_class(series):
    """
    Convert a continuous sensitivity variable to ranked vulnerability class.
    """
    rank = series.rank(pct=True)

    return pd.cut(
        rank,
        bins=[0, 0.25, 0.50, 0.75, 0.90, 1.00],
        labels=class_order,
        include_lowest=True
    )

map_gdf["peak_vulnerability_class"] = make_rank_class(
    map_gdf["peak_q90_plus10"]
)

map_gdf["volume_vulnerability_class"] = make_rank_class(
    map_gdf["volume_q90_plus10"]
)

map_gdf["combined_vulnerability_class"] = make_rank_class(
    map_gdf["combined_vulnerability_score"]
)

type_order = [
    "High peak and volume",
    "Peak-dominated",
    "Volume-dominated",
    "Mixed / moderate"
]

type_colors = {
    "High peak and volume": "#7b3294",
    "Peak-dominated": "#1f78b4",
    "Volume-dominated": "#e66101",
    "Mixed / moderate": "#bdbdbd",
}


from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd


def add_circular_bar_legend(
    ax,
    counts,
    colors,
    order,
    title,
    loc=(0.60, 0.05, 0.36, 0.36),
    fontsize=5.5
):
    """
    Add a circular bar-chart legend inside a map axis.

    counts: Series indexed by class/type labels
    colors: dict mapping labels to colors
    order: list controlling legend order
    loc: (x0, y0, width, height) in axis fraction
    """

    counts = counts.reindex(order).fillna(0)

    inset = ax.inset_axes(loc, projection="polar")

    values = counts.values.astype(float)
    labels = counts.index.tolist()

    if values.max() == 0:
        values_scaled = values
    else:
        values_scaled = values / values.max()

    n = len(labels)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n * 0.72

    # Base radius and bar height
    bottom = 0.35
    heights = 0.55 * values_scaled

    for t, h, label, value in zip(theta, heights, labels, values):
        inset.bar(
            t,
            h,
            width=width,
            bottom=bottom,
            color=colors[label],
            edgecolor="white",
            linewidth=0.35
        )

        # label around the circle
        angle_deg = np.degrees(t)
        rotation = angle_deg if angle_deg <= 180 else angle_deg - 180

        inset.text(
            t,
            bottom + h + 0.12,
            f"{label}\n{int(value)}",
            ha="center",
            va="center",
            fontsize=fontsize,
            rotation=rotation,
            rotation_mode="anchor"
        )

    inset.text(
        0,
        0,
        title,
        ha="center",
        va="center",
        fontsize=fontsize + 0.3
    )

    inset.set_ylim(0, 1.15)
    inset.set_axis_off()
    
    
def plot_categorical_map(
    gdf,
    ax,
    column,
    order,
    colors,
    alpha=0.92,
    zorder=10
):
    """
    Plot categorical polygons manually so colors always match labels.
    """

    for label in order:
        subset = gdf[gdf[column] == label]

        if len(subset) == 0:
            continue

        subset.plot(
            ax=ax,
            facecolor=colors[label],
            edgecolor="none",
            linewidth=0,
            alpha=alpha,
            zorder=zorder
        )
        
        
fig, axes = plt.subplots(
    1,
    4,
    sharex=True,
    sharey=True,
    figsize=(9.5, 8.5),
    dpi=600,
    constrained_layout=False
)

panel_titles = [
    "(a) Peak sensitivity",
    "(b) Volume sensitivity",
    "(c) Combined vulnerability",
    "(d) Vulnerability type"
]

# --------------------------------------------------
# Keep your original background unchanged
# --------------------------------------------------

for ax, title in zip(axes, panel_titles):
    _subplot_setup(ax, extent["DK"], "1.0")

    DK.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=0.35,
        zorder=5
    )

    catchment_polygons.plot(
        ax=ax,
        facecolor="none",
        edgecolor="gray",
        linewidth=0.05,
        zorder=6
    )

    ax.set_title(title, fontsize=8)


# --------------------------------------------------
# Panel a: peak-discharge sensitivity class
# --------------------------------------------------

plot_categorical_map(
    gdf=map_gdf,
    ax=axes[0],
    column="peak_vulnerability_class",
    order=class_order,
    colors=vulnerability_colors,
    alpha=0.92,
    zorder=10
)

peak_counts = (
    map_gdf["peak_vulnerability_class"]
    .value_counts()
    .reindex(class_order)
    .fillna(0)
)

add_circular_bar_legend(
    ax=axes[0],
    counts=peak_counts,
    colors=vulnerability_colors,
    order=class_order,
    title="Peak\nrank",
    loc=(0.58, 0.54, 0.38, 0.38)
)


# --------------------------------------------------
# Panel b: flood-volume sensitivity class
# --------------------------------------------------

plot_categorical_map(
    gdf=map_gdf,
    ax=axes[1],
    column="volume_vulnerability_class",
    order=class_order,
    colors=vulnerability_colors,
    alpha=0.92,
    zorder=10
)

volume_counts = (
    map_gdf["volume_vulnerability_class"]
    .value_counts()
    .reindex(class_order)
    .fillna(0)
)

add_circular_bar_legend(
    ax=axes[1],
    counts=volume_counts,
    colors=vulnerability_colors,
    order=class_order,
    title="Volume\nrank",
    loc=(0.58, 0.54, 0.38, 0.38)
)


# --------------------------------------------------
# Panel c: combined vulnerability rank
# --------------------------------------------------

plot_categorical_map(
    gdf=map_gdf,
    ax=axes[2],
    column="combined_vulnerability_class",
    order=class_order,
    colors=vulnerability_colors,
    alpha=0.92,
    zorder=10
)

combined_counts = (
    map_gdf["combined_vulnerability_class"]
    .value_counts()
    .reindex(class_order)
    .fillna(0)
)

add_circular_bar_legend(
    ax=axes[2],
    counts=combined_counts,
    colors=vulnerability_colors,
    order=class_order,
    title="Combined\nrank",
    loc=(0.58, 0.54, 0.38, 0.38)
)


# --------------------------------------------------
# Panel d: dominant vulnerability type
# --------------------------------------------------

plot_categorical_map(
    gdf=map_gdf,
    ax=axes[3],
    column="vulnerability_type",
    order=type_order,
    colors=type_colors,
    alpha=0.92,
    zorder=10
)

type_counts = (
    map_gdf["vulnerability_type"]
    .value_counts()
    .reindex(type_order)
    .fillna(0)
)

add_circular_bar_legend(
    ax=axes[3],
    counts=type_counts,
    colors=type_colors,
    order=type_order,
    title="Type",
    loc=(0.54, 0.54, 0.42, 0.42)
)


# --------------------------------------------------
# Redraw outlines on top for clean map boundaries
# --------------------------------------------------

for ax in axes:
    DK.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=0.35,
        zorder=20
    )

    catchment_polygons.boundary.plot(
        ax=ax,
        edgecolor="black",
        linewidth=0.015,
        alpha=0.20,
        zorder=21
    )

plt.tight_layout()
plt.show()

