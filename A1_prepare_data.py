# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:44:23 2026

@author: juliu
"""


'''
1. dem
2. landuse
3. soil and geology
2. DK-model, dtp and soil moisture

'''
import pickle
import os 
import numpy as np 
import pandas as pd 
import geopandas as gpd
from shapely.ops import transform
from shapely.validation import make_valid
from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xarray as xr
import rasterio
import rioxarray as rxr
from rasterio.transform import from_origin
import geopandas as gpd
from shapely import force_2d
from scipy import ndimage

from A0_detect_flood_events import load_camels_info, load_camels_dynamic, load_dkm_dynamics

# CAMELS-dk
stations, catchments, attributes = load_camels_info()

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


def calculate_dem_features(
    dem,
    pixel_size,
    nodata=None,
    flat_slope_threshold_deg=2,
    lowland_percentile=25):
    """
    Calculate simple DEM-derived terrain features.

    Features:
    - mean elevation
    - elevation standard deviation
    - mean slope
    - flat-area fraction
    - elevation range
    - terrain ruggedness
    - lowland fraction

    Parameters
    ----------
    dem : 2D numpy array
        DEM array. Areas outside the region should be np.nan or nodata.
    pixel_size : float
        DEM cell size in meters.
    nodata : float, optional
        Nodata value.
    prefix : str
        Prefix for output feature names, e.g. 'catch', 'valley', 'station'.
    flat_slope_threshold_deg : float
        Slope threshold for defining flat areas, in degrees.
    lowland_percentile : float
        Elevation percentile used to define lowland cells.

    Returns
    -------
    features : dict
        Dictionary of terrain features.
    """

    z = np.array(dem, dtype=float)

    if nodata is not None:
        z[z == nodata] = np.nan

    valid = np.isfinite(z)

    if valid.sum() == 0:
        return {
            f"elev_mean": np.nan,
            f"elev_std": np.nan,
            f"slope_mean_deg": np.nan,
            f"flat_fraction": np.nan,
            f"elev_range": np.nan,
            f"ruggedness": np.nan,
            f"lowland_fraction": np.nan,
        }

    # -----------------------------
    # Elevation features
    # -----------------------------
    elev = z[valid]

    elev_mean = np.nanmean(elev)
    elev_std = np.nanstd(elev)
    elev_range = np.nanmax(elev) - np.nanmin(elev)

    # -----------------------------
    # Fill NaNs for slope/ruggedness calculation
    # -----------------------------
    z_filled = z.copy()

    if np.isnan(z_filled).any():
        nan_mask = ~np.isfinite(z_filled)

        indices = ndimage.distance_transform_edt(
            nan_mask,
            return_distances=False,
            return_indices=True
        )

        z_filled = z_filled[tuple(indices)]

    # -----------------------------
    # Slope
    # -----------------------------
    dz_dy, dz_dx = np.gradient(z_filled, pixel_size, pixel_size)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    slope_deg[~valid] = np.nan

    slope_values = slope_deg[valid]

    slope_mean = np.nanmean(slope_values)
    flat_fraction = np.nanmean(slope_values < flat_slope_threshold_deg)

    # -----------------------------
    # Terrain ruggedness
    # Simple ruggedness = mean absolute difference
    # between each cell and its 8-neighbor mean
    # -----------------------------
    kernel = np.ones((3, 3), dtype=float)
    kernel[1, 1] = 0

    neighbor_mean = ndimage.convolve(z_filled, kernel, mode="nearest") / 8.0

    ruggedness_array = np.abs(z_filled - neighbor_mean)
    ruggedness_array[~valid] = np.nan

    ruggedness = np.nanmean(ruggedness_array[valid])

    # -----------------------------
    # Lowland fraction
    # Fraction of cells below chosen elevation percentile
    # -----------------------------
    lowland_cutoff = np.nanpercentile(elev, lowland_percentile)
    lowland_fraction = np.nanmean(elev <= lowland_cutoff)

    features = {
        f"elev_mean": elev_mean,
        f"elev_std": elev_std,
        f"slope_mean_deg": slope_mean,
        f"flat_fraction": flat_fraction,
        f"elev_range": elev_range,
        f"ruggedness": ruggedness,
        f"lowland_fraction": lowland_fraction,
    }

    return features

def _load_10m_DEM() -> xr.Dataset:
    #read in 10m DEM data
    dataPath = r"\\geodata.geus.dk\DKmodel_users\FloodWarning\LSTM_postprocessing\CAMELS-DK\Level_1\topography\DEM_10m.tif"
    DEM_10m  = rxr.open_rasterio(dataPath, masked=True)
    if "band" in DEM_10m.dims and DEM_10m.sizes["band"] == 1:
        DEM_10m = DEM_10m.squeeze("band", drop=True)
    DEM_10m = DEM_10m.to_dataset(name="top")  # or any variable name you want
    # DEM_10m["top"] = DEM_10m["top"].where(DEM_10m["top"] != -99, DEM_10m['top'].nodatavals)
    
    return DEM_10m





def calculate_landuse_fractions(landuse):
    """
    Calculate land-use class fractions from a categorical raster.

    Parameters
    ----------
    landuse : 2D numpy array
        Land-use raster array clipped/masked to the target region.
        Areas outside the region should be np.nan or nodata.
    nodata : int
        Nodata class code.
    prefix : str
        Prefix for output names, e.g. 'catch', 'valley', 'station'.

    Returns
    -------
    features : dict
        Land-use fractions for each class.
    """

    class_map = {
        0: "other",
        1: "urban_impervious_artificial",
        2: "agriculture_annual",
        3: "agriculture_permanent_or_other",
        4: "grassland",
        5: "forest",
        6: "wetland",
        7: "open_water",
    }

    arr = np.array(landuse)

    # valid cells: not nodata and finite
    valid = np.isfinite(arr)

    n_valid = valid.sum()

    features = {
        f"landuse_valid_cells": int(n_valid)
    }

    if n_valid == 0:
        for code, name in class_map.items():
            features[f"frac_{name}"] = np.nan
        return features

    valid_values = arr[valid]

    for code, name in class_map.items():
        features[f"frac_{name}"] = np.mean(valid_values == code)

    return features


def _load_basemap() -> xr.Dataset:
    # get land use
    ## read basemap 2021 data
    basemap04Path = r'\\geodata.geus.dk\DKmodel_users\FloodWarning\LSTM_postprocessing\CAMELS-DK\Level_1\landuse\Basemap04_2021'
    basemap04Name = 'lu_agg_2021.tif'
    basemap04Data = rxr.open_rasterio(os.path.join(basemap04Path, basemap04Name), masked=True)
    
    if "band" in basemap04Data.dims and basemap04Data.sizes["band"] == 1:
        basemap04Data = basemap04Data.squeeze("band", drop=True)
    basemap04Data = basemap04Data.to_dataset(name="landuse")  # or any variable name you want
    
    # classify

    da = basemap04Data["landuse"]
    
    # Output class codes
    # 0 = other
    # 1 = urban_impervious/artificial
    # 2 = agriculture_annual
    # 3 = agriculture_permanent_or_other
    # 4 = grassland
    # 5 = forest
    # 6 = wetland
    # 7 = open_water
    # 255 = nodata
    
    hydro_lu = xr.full_like(da, 0, dtype="uint8")
    
    hydro_lu = xr.where((da >= 110000) & (da < 200000), 1, hydro_lu)
    
    hydro_lu = xr.where(da.isin([211000, 212000]), 2, hydro_lu)
    hydro_lu = xr.where(da.isin([220000, 230000]), 3, hydro_lu)
    
    hydro_lu = xr.where(
        da.isin([311000, 312000, 321000, 321220, 322000, 322220]),
        4,
        hydro_lu
    )
    
    hydro_lu = xr.where(da.isin([411000, 412000, 420000]), 5, hydro_lu)
    
    # No clear wetland code in your unique values.
    # If later you see 700000 or 7xxxxx codes, classify them as wetland.
    hydro_lu = xr.where((da >= 700000) & (da < 800000), 6, hydro_lu)
    
    hydro_lu = xr.where(da == 800000, 7, hydro_lu)
    
    hydro_lu = hydro_lu.where(~np.isnan(da), 255)
    hydro_lu = xr.where(da == 999999, 255, hydro_lu)
    
    hydro_lu.name = "hydrological_landuse_class"
    
    hydro_lu = hydro_lu.astype("float32")
    hydro_lu = hydro_lu.where(hydro_lu != 255)

    return hydro_lu

def _load_10m_dtp():
    # Load DTP rasters
    winter = rxr.open_rasterio(r"\\geodata.geus.dk\DKmodel_users\FloodWarning\GWH_emulator\Hydro03_E_temporary\juliu\GWH_emulator\Dataset\10m_dtp\Winter_predict.tif", masked=True)
    summer = rxr.open_rasterio(r"\\geodata.geus.dk\DKmodel_users\FloodWarning\GWH_emulator\Hydro03_E_temporary\juliu\GWH_emulator\Dataset\10m_dtp\Summer_predict.tif", masked=True)

    # Drop the band dimension if present
    if "band" in winter.dims and winter.sizes["band"] == 1:
        winter = winter.squeeze("band", drop=True)
    if "band" in summer.dims and summer.sizes["band"] == 1:
        summer = summer.squeeze("band", drop=True)

    # Convert to datasets
    winter = winter.to_dataset(name="dtp")
    summer = summer.to_dataset(name="dtp")
    
    if winter.rio.crs is None:
        winter = winter.rio.write_crs("EPSG:25832", inplace=False)
    if summer.rio.crs is None:
        summer = summer.rio.write_crs(winter.rio.crs, inplace=False)
    
    # Compute yearly mean
    yearly = (winter["dtp"] + summer["dtp"]) / 2.0
    yearly = yearly.to_dataset(name="dtp")

    return summer, winter, yearly



grid_folder = r'\\geodata.geus.dk\DKmodel_users\FloodWarning\LSTM_postprocessing\CAMELS-DK\Level_1\Soil'
data_dict   = {}
for var_name in ['claynor', 'fsandno', 'gsandno']:
    for order in ['_30', '_60', '_100', '_200']:
        if order=='_30': oder_e='a'
        if order=='_60': oder_e='b'
        if order=='_100': oder_e='c'
        if order=='_200': oder_e='d'
        data_dict[var_name+order] = rxr.open_rasterio(os.path.join(grid_folder, oder_e+var_name+'.tif'),masked=True)
        
def _load_soil(accum_shp):
    dict_pct = {}
    for var_name in ['claynor', 'fsandno', 'gsandno']:
        for order in ['_30', '_60', '_100', '_200']:
            id15_colname = 'pct_'+var_name+order
            ds = data_dict[var_name+order].rio.clip(accum_shp.geometry, from_disk=True, all_touched=True)
            dict_pct[id15_colname] = ds.mean().values.item()
    
    return dict_pct


geoPath = r'\\geodata.geus.dk\DKmodel_users\FloodWarning\LSTM_postprocessing\CAMELS-DK\Level_1\Geology'
dk_kalk_dyb            = rxr.open_rasterio(os.path.join(geoPath, 'dk_kalk_dyb.tif'), masked=True)
dk_topmag_akku         = rxr.open_rasterio(os.path.join(geoPath, 'dk_topmag_akku.tif'), masked=True)
dk_ler2mag_1m_exj25200 = rxr.open_rasterio(os.path.join(geoPath, 'dk_ler2mag_1m_exj25200.tif'), masked=True)
dk_ler_ts_1m           = rxr.open_rasterio(os.path.join(geoPath, 'dk_ler_ts_1m.tif'), masked=True)
dk_mag2ler_1m          = rxr.open_rasterio(os.path.join(geoPath, 'dk_mag2ler_1m.tif'), masked=True)

def _load_geology(accum_shp):
    
    dict_pct = {}
    for geoname in ['dk_kalk_dyb', 
                    'dk_topmag_akku', 
                    'dk_ler2mag_1m_exj25200',
                    'dk_ler_ts_1m', 
                    'dk_mag2ler_1m']:
        id15_colname = geoname
        
        if geoname=='dk_kalk_dyb':            ds = dk_kalk_dyb.rio.clip(accum_shp.geometry, from_disk=True, all_touched=True)
        if geoname=='dk_topmag_akku':         ds = dk_topmag_akku.rio.clip(accum_shp.geometry, from_disk=True, all_touched=True)
        if geoname=='dk_ler2mag_1m_exj25200': ds = dk_ler2mag_1m_exj25200.rio.clip(accum_shp.geometry, from_disk=True, all_touched=True)
        if geoname=='dk_ler_ts_1m':           ds = dk_ler_ts_1m.rio.clip(accum_shp.geometry, from_disk=True, all_touched=True)
        if geoname=='dk_mag2ler_1m':          ds = dk_mag2ler_1m.rio.clip(accum_shp.geometry, from_disk=True, all_touched=True)

        dict_pct[id15_colname] = ds.mean().values.item()

    return dict_pct



def pca_top10(catchment_alpha):
    '''
    PCA on the unique catchment-level AlphaEarth table, 
    not on the full event table.

    Returns
    -------
    dataset.

    '''
    alpha_cols = [c for c in catchment_alpha.columns if c.startswith("A")]

    catch_alpha = catchment_alpha[['catch_id'] + alpha_cols].drop_duplicates('catch_id')
    
    scaler = StandardScaler()
    X_alpha_scaled = scaler.fit_transform(catch_alpha[alpha_cols])
    
    pca = PCA(n_components=10, random_state=0)
    pcs = pca.fit_transform(X_alpha_scaled)
    
    for i in range(10):
        catch_alpha[f"AE_PC{i+1}"] = pcs[:, i]
    
    return catch_alpha

# DEM_10m = _load_10m_DEM()
# landuse = _load_basemap()
# summer, winter, yearly = _load_10m_DEM()



#%% statics
# Load ID15 catchments
id15_shp = gpd.read_file(r"\\geodata.geus.dk\DKmodel_users\FloodWarning\Literature_survey\GWH_flood\CAMELS-DK\Shapefile\CAMELS_DK_3330_catchment_boundaries.shp")

# Load catchments
stations, catchments, attributes = load_camels_info()
catchments = catchments.drop(index=218)

# prepare 3 scales
# scale 1. cathcment scale
catchments = catchments.copy()

# scale 2. downstream ID15 basins
DS_basin = id15_shp[id15_shp['catch_id'].isin(stations["catch_id"].to_list())]

# scale 3. river velley
# River_valley = gpd.read_file(r"\\geodata.geus.dk\DKmodel_users\FloodWarning\Literature_survey\Causal_ML\data\shp\VANDLOEBSMIDTE_2025.shp")
# River_valley = River_valley.buffer(500) # 500 m buffer

dataset_ee = {}
for st in stations['catch_id']:
    id15 = int(st)
    print('id15:, ', id15)
    shp_catch = catchments[catchments["catch_id"] == id15]
    shp_downs = DS_basin[DS_basin["catch_id"] == id15]
    shp_valley= River_valley.clip(shp_catch)
    shp_valley= gpd.GeoDataFrame(geometry=shp_valley, crs=shp_valley.crs).dissolve()
    
    df_ee_catch = get_ee_features(shp_catch)
    df_ee_downs = get_ee_features(shp_downs)
    df_ee_valley= get_ee_features(shp_valley)
    
    dataset_ee[st] = {}
    dataset_ee[st]['shp_catch'] = shp_catch
    dataset_ee[st]['shp_downs'] = shp_downs
    dataset_ee[st]['shp_valley']= shp_valley
    
    dataset_ee[st]['df_ee_catch'] = df_ee_catch
    dataset_ee[st]['df_ee_downs'] = df_ee_downs
    dataset_ee[st]['df_ee_valley']= df_ee_valley
    


# save
with open(r"data/alphaearth/shp_df_ee_features.pkl", "wb") as f:
    pickle.dump(dataset_ee, f, protocol=pickle.HIGHEST_PROTOCOL)

# load
# with open(r"data/alphaearth/shp_df_ee_features.pkl", "rb") as f:
#     dataset_ee = pickle.load(f)

#%% dynamics
def point_in_valid_area(point, valid_mask):
    x = point.x.values[0]
    y = point.y.values[0]

    # Select nearest xarray grid cell
    value = valid_mask.sel(x=x, y=y, method='nearest').item()

    return bool(value)

with open(r"data/alphaearth/shp_df_ee_features.pkl", "rb") as f:
    dataset_ee = pickle.load(f)
    
    
dynamic = {}
for DKM_id in range(1, 7):
    dtp, wcr = load_dkm_dynamics(DKM_id, jumpfilled=True)
    dtp, wcr = dtp.rio.write_crs(stations.crs), wcr.rio.write_crs(stations.crs)
    valid_mask = dtp['dtp'].mean(dim='time').notnull()
    
    for st in dataset_ee.keys():
        point = stations[stations['catch_id'] == st].geometry
        if point_in_valid_area(point, valid_mask):
            print(f'Found st: {st} in DKM_id {DKM_id}')
            shp_catch  = dataset_ee[st]['shp_catch'] 
            shp_downs  = dataset_ee[st]['shp_downs'] 
            shp_valley = dataset_ee[st]['shp_valley']
            
            dynamic[st] = {}
            dynamic[st]['dtp_catch'] = dtp.rio.clip(shp_catch.geometry, all_touched =True, drop=True).mean(dim=['x', 'y']).to_dataframe()
            dynamic[st]['dtp_downs'] = dtp.rio.clip(shp_downs.geometry, all_touched =True, drop=True).mean(dim=['x', 'y']).to_dataframe()
            dynamic[st]['dtp_valley']= dtp.rio.clip(shp_valley.geometry,all_touched =True, drop=True).mean(dim=['x', 'y']).to_dataframe()
            
            
            dynamic[st]['wcr_catch'] = wcr.rio.clip(shp_catch.geometry, all_touched =True, drop=True).mean(dim=['x', 'y']).to_dataframe()
            dynamic[st]['wcr_downs'] = wcr.rio.clip(shp_downs.geometry, all_touched =True, drop=True).mean(dim=['x', 'y']).to_dataframe()
            dynamic[st]['wcr_valley']= wcr.rio.clip(shp_valley.geometry,all_touched =True, drop=True).mean(dim=['x', 'y']).to_dataframe()

    # save
    with open(rf"data/alphaearth/shp_df_dtp_wcr_DKM{DKM_id}.pkl", "wb") as f:
        pickle.dump(dynamic, f, protocol=pickle.HIGHEST_PROTOCOL)    