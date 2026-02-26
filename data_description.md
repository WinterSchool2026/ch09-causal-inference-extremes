### 🗂️ Data Sources

You will work with a multimodal and harmonized dataset. It includes data from several sources. Data has been deseasoned (remove seasonal patterns, representing anomalies) and standardized (z-scores).

- [EDID](https://drought.emergency.copernicus.eu/tumbo/edid/about): European Drought Impact Database (EDID) – A comprehensive dataset tracking drought impacts in Europe 
- [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=overview): ERA5 monthly averaged data on single levels from 1940 to present
- [ERA5-Drought](https://cds.climate.copernicus.eu/datasets/derived-drought-historical-monthly?tab=overview): ERA5–Drought is a global reconstruction of drought indices from 1940 to present.
- [GLEAM v4.2b](https://www.gleam.eu/): GLEAM provides data of the different components of land evaporation.
- [EDO](https://drought.emergency.copernicus.eu/tumbo/edo/download/): Drought Observatory of the Copernicus Emergency Management Service
- [MODIS](https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod13q1-061): The Moderate Resolution Imaging Spectroradiometer (MODIS), measures visible and infrared radiation and obtaining data that are being used to derive products ranging from vegetation, land surface cover, and ocean chlorophyll fluorescence to cloud and aerosol properties, fire occurrence, snow cover on the land, and sea ice cover on the oceans.
- [WORLDPOP](https://www.worldpop.org/): WorldPop develops peer-reviewed research and methods for the construction of open and high-resolution geospatial data on population distributions, demographic and dynamics.
- [COP30-DEM](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM): The Copernicus DEM is a Digital Surface Model (DSM) that represents the surface of the Earth including buildings, infrastructure and vegetation.
- [Global-HAND](https://www.researchgate.net/publication/301559649_Global_30m_Height_Above_the_Nearest_Drainage): The Height Above the Nearest Drainage (HAND), a digital elevation model normalized using the nearest drainage is used for hydrological and more general purpose applications, such as hazard mapping, landform classification, and remote sensing.
- [Geomorpho90m](https://portal.opentopography.org/dataspace/dataset?opentopoID=OTDS.012020.4326.1): A global dataset comprising of 26 geomorphometric features derived from the MERIT-DEM.
- [Microsoft Roads](https://github.com/microsoft/RoadDetections/tree/main?tab=readme-ov-file): Roads around the world.
- [CCI-LC](https://www.esa-landcover-cci.org/?q=node/164): fully automated global land cover mapping at 300m resolution.
- [European Soil Database Derived data](https://esdac.jrc.ec.europa.eu/content/european-soil-database-derived-data): A number of layers for soil properties have been created based on data from the European Soil Database in combination with data from the Harmonized World Soil Database (HWSD) and Soil-Terrain Database (SOTER).
- [HydroSHEDS](https://www.hydrosheds.org/): suite of global digital data layers in support of hydro-ecological research and applications worldwide.
- [GAEZ (FAO)](https://data.apps.fao.org/gaez/?lang=en): Agro-Ecological Zones (AEZ), relies on well-established land evaluation principles to assess natural resources for finding suitable agricultural land utilization options
- [NOAA/PDL](https://psl.noaa.gov/data/timeseries/month/): Monthly Climate/Ocean Indices (Time-Series) at the Physical Sciences Laboratory (PSL)


### 🌍 Description of variables

### Target / Outcome:

| **Name**                                                                              | **Variable name** | **Unit**                              | *Source* |
| ------------------------------------------------------------------------------------- | ----------------- | ------------------------------------- | -------- |
| Drought impact in the agriculture sector (severe and extreme impacts, at least during 60 days) | DI_agri_extreme_M7 | Binary | EDID |


### Climate, environmental and socio-economic factors

### - Potential Treatments:

| **Name**                                               | **Variable name**             | **Unit**      | *Source*                                                                                                         |
| ------------------------------------------------------ | ----------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------- |
| Standard Precipitation Evapotranspitation Index (1,3,6,12,24,38,48M) | SPEI | (-4,4) | ERA5-Drought|
| Standard Precipitation Index (1,3,6,12,24,38,48M) | SPI | (-4,4) | ERA5-Drought |
| Combine Drought Index | CDI | categories (1,2,3) | EDO |
| Soil Moisture Anomaly | SMA | categories (1,2,3) | EDO |


### - Potential confounding and moderators:

| **Name**                                               | **Variable name**             | **Unit**      | *Source*                                                                                                         |
| ------------------------------------------------------ | ----------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------- |
| Evaporation (anomaly) | e_ds | m of water equivalent | ERA5|
| Potential evaporation (anomaly) | pev_ds | m | ERA5 |
| Runoff (anomaly) | ro_ds | m | ERA5 |
| Surface runoff  (anomaly) | sro_ds | m | ERA5 | 
| Surface latent heat flux (anomaly) | slhf_ds | J m**-2 | ERA5|
| Surface net solar radiation (anomaly) | ssr_ds | J m**-2 | ERA5|
| Surface solar radiation downwards (anomaly) | ssrd_ds | J m**-2 | ERA5 |
| Surface net thermal radiation (anomaly) | str_ds | J m**-2 | ERA5 |
| Surface thermal radiation downwards (anomaly) | strd_ds | J m**-2 | ERA5 |
| Total precipitation (anomaly) | tp_ds | m | ERA5 | 
| Neutral wind at 10 m u-component (anomaly) | u10_ds | m s**-1 | ERA5 | 
| Neutral wind at 10 m v-component (anomaly) | v10_ds | m s**-1 | ERA5 | 
| 2 metre dewpoint temperature (anomaly) | d2m_ds | K| ERA5|
| Mean sea level pressure (anomaly) | msl_ds | Ps | ERA5 |
| Sea surface temperature (anomaly) | sst_ds | K | ERA5 |
| Surface pressure (anomaly) | sp_ds | Pa | ERA5 |
| Skin temperature (anomaly) | skt_ds | K | ERA5 | 
| Total column water vapour (anomaly) | tcwv_ds | kg m**-2 | ERA5 | 
| Total column water (anomaly) | tcw_ds | kg m**-2 | ERA5 | 
| Air density over the oceans (anomaly) | rhoao_ds | kg m**-3 | ERA5 |
| Actual Evaporation (anomaly) | E_gleam_ds | mm d^-1 | GLEAM v4.2b |
| Potential Evaporation (anomaly) | Ep_gleam_ds | mm d^-1 | GLEAM v4.2b |
| Aerodynamic component of Potential Evaporation (anomaly) | Ep_aero_gleam_ds | mm d^-1 | GLEAM v4.2b |
| Radiative component of Potential Evaporation (anomaly) | Ep_rad_gleam_ds | mm d^-1 | GLEAM v4.2b |
| Interception loss (anomaly) | Ei_gleam_ds | mm d^-1 | GLEAM v4.2b |
| Bare-soil Evaporation (anomaly) | Eb_gleam_ds | mm d^-1 | GLEAM v4.2b |
| Snow sublimation (anomaly) | Es_gleam_ds | mm d^-1 | GLEAM v4.2b |
| Transpiration (anomaly) | Et_gleam_ds | mm d^-1 | GLEAM v4.2b |
| Open-water evaporation (anomaly) | Ew_gleam_ds | mm d^-1 | GLEAM v4.2b |
| Surface condensation (anomaly) | Ec_gleam_ds | mm d^-1 | GLEAM v4.2b |
| Evaporative stress factor (anomaly) | S_gleam_ds | - | GLEAM v4.2b |
| Root-zone soil moisture (anomaly) | SMrz_gleam_ds | m^3 d^-3 | GLEAM v4.2b |
| Surface Soil Moisture (anomaly) | SMs_gleam_ds | m^3 d^-3 | GLEAM v4.2b |
| Sensible heat flux (anomaly) | H_gleam_ds | W d^-2 | GLEAM v4.2b |
| Daytime Land Surface Temperature (anomaly) | lst_day_ds | K | MODIS |
| Nighttime Land Surface Temperature (anomaly) | lst_night_ds | K | MODIS |
| Normalized Difference Vegetation Index (anomaly) | ndvi_ds | - | MODIS |
| Normalized Difference Water Index (anomaly) | ndwi_ds | - | MODIS |
| Mean Population | pop | - | Worldpop |
| Mean elevation | dem | m | COP30 |
| Mean height above nearest drainage | hand | m | Global-HAND|
| Mean Flow accumulation - number of upstream grid cells | acc | - | HydroSHEDS |
| Mean Stream Power Index | spi | - |Geomorpho90m|
| Mean Compound Topographic Index | cti | - | Geomorpho90m|
| Road density | road | km km ^-2 | Roads|
| Land cover percentage | lc | (%) | CCI-LC |
| Irrigated crops | agri_irri | (%) | CCI-LC |
| Rainfed crops | agri_rain | (%) | CCI-LC |
| Mixed crops with vegetation | agri_mix | (%) | CCI-LC|
| Topsoil Clay content | soil_clay | (%) | European Soil Database Derived data |
| Topsoil Organic Carbon content | soil_oc | (%) | European Soil Database Derived data |
| Depth available to roots | soil_roots | cm | European Soil Database Derived data |
| Topsoil Sand content | soil_sand | (%) | European Soil Database Derived data |
| Subsoil Total available water content from PTF | soil_tawc | mm | European Soil Database Derived data |
| Mean total precipitation for the basin level 8 (anomaly) | tp_basin_mean_ds | m | ERA5 & HydroSHEDS |


### - Teleconnections

Teleconnections are large-scale, persistent atmospheric pressure and circulation patterns that link weather anomalies across widely separated regions, often thousands of kilometers apart. These recurring patterns, driven by ocean-atmosphere interactions, allow climate conditions in one area to influence another over weeks or months. Key examples include the El Niño-Southern Oscillation (ENSO), North Atlantic Oscillation (NAO), and Pacific-North American Pattern (PNA)

| **Name**                                       | **Variable name**     | **Unit**     | *Source*        |
| ---------------------------------------------- | --------------------- | ------------ | --------------- |
| Arctic Oscillation                             | ao_long               | mb           | NOAA/CPC        |
| Bivariate EnSo Timeseries                      | censo                 | std          | NOAA/PSL        |
| Eastern Atlantic                               | ea                    | mb           | NOAA/CPC        |
| North Atlantic Oscillation                     | nao_long              | mb           | NOAA/NGDC       |
| Niño 3.4 (HadISST)                             | nino34_long_anom      | Cº           | NOAA/PSL        |
| NOAA Global Average Land Temperature Anomalies | noaa_globaltmp_comb   | Cº           | NOAA/NCEI       |
| Pacific Decadal Oscillation PSL                | pdo_timeseries_sstens | Cº           | U of Washington |
| Pacific North American Index                   | pna                   | mb           | NOAA/CPC        |
| Southern Oscillation Index                     | soi_long              | standardized | UEA CRU         |
| West Pacific Index                             | wp                    | mb           | NOAA/CPC        |


### Climate zones and hydrological basins (regions)

| **Name**                            | **Variable name**   |   *Source*     |
| ----------------------------------- | ------------------- | ------------ | 
| Hydrological basins (level 2) | basin | HydroSHEDS|
| Hydrological basins (level 3) | basin | HydroSHEDS |
| Agro-ecological zones (Koeppen-Geiger climate classification)  | KG2 |  GAEZ (FAO) |
| Agro-ecological zones (Thermal regimes) | thz | GAEZ (FAO) |