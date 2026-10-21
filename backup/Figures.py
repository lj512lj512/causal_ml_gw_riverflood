# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:31:53 2026

@author: juliu
"""

import os
import pandas as pd 
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import sys
sys.path.append(r'\\geodata.geus.dk\DKmodel_users\FloodWarning\LSTM_postprocessing\CAMELS-DK\Level_1\DKM_simulations')
import RS_dfs_mikeio
import xarray as xr 

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

#%% dk

# 1. prepare shapefile
DK  = gpd.read_file(r'\\geodata.geus.dk\HOME\Discharge_correction_with_LSTM\Shapfile\DKDomains2013\kort10_land_DK.shp')
BH  = gpd.read_file(r'\\geodata.geus.dk\HOME\Discharge_correction_with_LSTM\Shapfile\DKDomains2013\kort10_land_BH.shp')


dataFolder = r'\\geodata.geus.dk\DKmodel_users\FloodWarning\Literature_survey\GWH_flood\CAMELS-DK'
stations   = gpd.read_file(os.path.join(dataFolder, 
                                      'Shapefile', 
                                      'CAMELS_DK_304_gauging_stations.shp'))
catchments = gpd.read_file(os.path.join(dataFolder, 
                                      'Shapefile', 
                                      'CAMELS_DK_304_gauging_catchment_boundaries.shp'))
id15_shp   = gpd.read_file(os.path.join(dataFolder, 
                                      'Shapefile', 
                                      'CAMELS_DK_3330_catchment_boundaries.shp'))
attributes = pd.read_csv(os.path.join(dataFolder, 
                                      'Attibutes', 
                                      'CAMELS_DK_topography.csv'))
attributes = attributes[attributes['gauge_record_pct'] > 80]

stations   = stations[stations['catch_id'].isin(attributes['catch_id'].to_list())]
catchments = catchments[catchments['catch_id'].isin(attributes['catch_id'].to_list())]

extent = {} # [[xmin, xmax], [ymin, ymax]]
extent['DK'] = [[439000, 739000], [6049000, 6404000]]
extent['BH'] = [[850000, 900000], [6104000, 6150000]]


# 2. prepare raster 
dem_folder = r'\\geodata.geus.dk\Dkmodel-hydro\HIP_Realtid\HIP_100m_500m_realtid_historical_1989_2024_newClimTS_precip'  
dtp_folder = r'\\geodata.geus.dk\DKmodel_users\FloodWarning\GWH_emulator\Hydro03_E_temporary\juliu\GWH_emulator\Dataset\DKM_sim'
resolution = 500

dem_all = []
dtp_all = []
for DKM_id in range(1, 7):
    datapath = os.path.join(dem_folder, f'DK{DKM_id}_HIPmodels', 'Result', f'DK{DKM_id}_HIP_{resolution}m')
    mname    = f'DK{DKM_id}_HIP_{resolution}m'
    dem      = RS_dfs_mikeio.extract_dfs_data(root=datapath,mname=mname, var = 'top').to_xarray()
    dtp_mean = xr.load_dataset(os.path.join(dtp_folder, f'DK{DKM_id}_HIP_{resolution}m_2DSZ_dtp_jumps_filled_1989_2024.nc'))
    
    dem_all.append(dem)
    dtp_all.append(dtp_mean.mean(dim='time') * (-1))

dem_all = xr.merge(dem_all)
dtp_all = xr.merge(dtp_all)

    

# # 3. make the plot
# fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11.5/2, 10), dpi=1200)

# # general figure setup
# # for i, axi in enumerate(fig.axes):
# #     _subplot_setup(axi, extent['DK'], '1.0')
# #     DK.plot(ax=axi, facecolor='none', edgecolor='black', linewidth=0.2) #add coastline
# #     if i == 0: 
# #         dem_all['Surface topography'].plot(ax=axi, )
# #         catchments.plot(ax=axi, facecolor='none', edgecolor='r', linewidth=0.2)
# #         stations.plot(ax=axi, facecolor='b', edgecolor='b', markersize=0.5)
        
#     # if i == 1: dtp_all.plot(ax=axi)
# for axi in ax:
#     _subplot_setup(ax, extent["DK"], "1.0")
#     DK.plot(ax=ax,facecolor="none",edgecolor="black",linewidth=0.35,zorder=5)
#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# -----------------------------
# Plot settings
# -----------------------------
fig, axes = plt.subplots(
    1, 2,
    sharex=True,
    sharey=True,
    figsize=(9.0, 8.5),
    dpi=600,
    constrained_layout=False
)

ax_dem, ax_dtp = axes

# Colormaps
dem_cmap = mcolors.LinearSegmentedColormap.from_list(
    "soft_dem",
    [
        "#f7fbff",  # very low elevation: almost white-blue
        "#e5f5e0",  # lowland: pale green
        "#c7e9c0",  # gentle green
        "#a1d99b",  # higher green
        "#d9c8a9",  # muted tan
        "#b8a07a",  # higher terrain brown
    ]
)
dtp_cmap = 'YlGnBu'
# Optional: set robust display ranges
# Adjust these depending on your actual values
dem_vmin = dem_all['Surface topography'].quantile(0.02).item() if hasattr(dem_all['Surface topography'], "quantile") else None
dem_vmax = dem_all['Surface topography'].quantile(0.98).item() if hasattr(dem_all['Surface topography'], "quantile") else None

dtp_vmin = dtp_all['dtp'].quantile(0.02).item() if hasattr(dtp_all['dtp'], "quantile") else None
dtp_vmax = dtp_all['dtp'].quantile(0.98).item() if hasattr(dtp_all['dtp'], "quantile") else None

# If DTP is anomaly and centered around zero, use symmetric limits
if dtp_vmin is not None and dtp_vmax is not None:
    dtp_abs = max(abs(dtp_vmin), abs(dtp_vmax))
    dtp_vmin, dtp_vmax = -dtp_abs, dtp_abs

# -----------------------------
# General axis setup
# -----------------------------
for ax in axes:
    _subplot_setup(ax, extent["DK"], "1.0")
    DK.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=0.35,
        zorder=5
    )

# -----------------------------
# Panel A: DEM
# -----------------------------
dem_im = dem_all['Surface topography'].plot(
    ax=ax_dem,
    cmap=dem_cmap,
    vmin=dem_vmin,
    vmax=dem_vmax,
    add_colorbar=False,
    zorder=1
)

# Catchment boundaries: clear but not too dominant
catchments.plot(
    ax=ax_dem,
    facecolor="none",
    edgecolor="#d73027",   # clear red
    linewidth=0.35,
    alpha=0.95,
    zorder=20
)

# Stations: visible white circles with black edge
stations.plot(
    ax=ax_dem,
    facecolor="white",
    edgecolor="black",
    linewidth=0.25,
    markersize=5,
    alpha=1.0,
    zorder=30
)


# Colorbar for DEM
divider = make_axes_locatable(ax_dem)
cax = inset_axes(
    ax_dem,
    width="100%",
    height="100%",
    bbox_to_anchor=(0.65, 0.52, 0.025, 0.45),  # x, y, width, height
    bbox_transform=ax_dem.transAxes,
    loc="lower left",
    borderpad=0
)
cb = fig.colorbar(dem_im,cax=cax,orientation="vertical")
cb.set_label("Elevation (m)", fontsize=15)
cb.ax.tick_params(labelsize=15, length=2)

# Optional: make colorbar background readable
cax.set_facecolor("white")
cax.patch.set_alpha(0.85)

# -----------------------------
# Panel B: groundwater depth / anomaly
# -----------------------------
dtp_im = dtp_all['dtp'].plot(
    ax=ax_dtp,
    cmap=dtp_cmap,
    vmin=0,
    vmax=10,
    add_colorbar=False,
    zorder=1
)


# Colorbar for DTP
divider = make_axes_locatable(ax_dtp)
cax = inset_axes(
    ax_dtp,
    width="100%",
    height="100%",
    bbox_to_anchor=(0.65, 0.52, 0.025, 0.45),  # x, y, width, height
    bbox_transform=ax_dtp.transAxes,
    loc="lower left",
    borderpad=0
)

cb = fig.colorbar(dtp_im,cax=cax,orientation="vertical")
cb.set_label("Groundwater depth (m)", fontsize=15)
cb.ax.tick_params(labelsize=15, length=2)

# Optional: make colorbar background readable
cax.set_facecolor("white")
cax.patch.set_alpha(0.85)

# -----------------------------
# Final layout
# -----------------------------
for ax in axes:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=15)
    ax.set_aspect("equal")

plt.subplots_adjust(left=0.03,right=0.98,top=0.94,bottom=0.08,wspace=0.04)
plt.show()
    
#%% case
# subplots of different scales
RIVERNAME = 'store_river'
catch_id  = 12430456
river_link= gpd.read_file(r"\\geodata.geus.dk\Dkmodel-hydro\HIP_Realtid\HIP_100m_500m_realtid_historical_1989_2024_newClimTS_precip\DK5_HIPmodels\Result\DK5_HIP_100m\DK5_HIP_100m_RiverLinks.shp")
ROI_catch = catchments[catchments['catch_id'] == catch_id]
ROI_ID15  = id15_shp[id15_shp['catch_id'] == catch_id]
ROI_station= stations[stations['catch_id'] == catch_id]
ROI_river = river_link.clip(ROI_catch).buffer(200).unary_union
ROI_river = gpd.GeoDataFrame(geometry=[ROI_river],crs=ROI_station.crs)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

fig, ax = plt.subplots(1, 1,figsize=(4, 2),dpi=1200)

# Colorblind-friendly colors
colors = {
    "Scale 1": "#999999",  # gray
    "Scale 2": "#D55E00",  # blue
    "Scale 3": "#0072B2",  # vermillion/orange "#D55E00"
}

alpha = 0.35
ROI_catch.plot(ax=ax,facecolor=colors["Scale 1"],edgecolor=colors["Scale 1"],
               linewidth=0.35,alpha=alpha,label="Scale 1",zorder=1)
ROI_ID15.plot(ax=ax,facecolor=colors["Scale 2"],edgecolor=colors["Scale 2"],
              linewidth=0.35,alpha=alpha,label="Scale 2",zorder=2)
ROI_river.plot(ax=ax,facecolor=colors["Scale 3"],edgecolor=colors["Scale 3"],
               linewidth=0.35,alpha=alpha,label="Scale 3",zorder=3)
ax.set_aspect("equal")
ax.axis("off")

# Rectangular legend patches
legend_handles = [
    Patch(
        facecolor=colors["Scale 1"],
        edgecolor=colors["Scale 1"],
        alpha=alpha,
        label="Scale 1"
    ),
    Patch(
        facecolor=colors["Scale 2"],
        edgecolor=colors["Scale 2"],
        alpha=alpha,
        label="Scale 2"
    ),
    Patch(
        facecolor=colors["Scale 3"],
        edgecolor=colors["Scale 3"],
        alpha=alpha,
        label="Scale 3"
    ),
]

ax.legend(handles=legend_handles,loc="upper right",frameon=False,fontsize=7,
          handlelength=1.2,
          handleheight=1.0,
          handletextpad=0.5,
          borderpad=0.2,
          labelspacing=0.4
)
ROI_station.plot(ax=ax,facecolor='r',
    edgecolor="r",
    markersize=5,
    zorder=1
)

plt.tight_layout(pad=0.05)

