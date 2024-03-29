# Renewable-Complementarity
The purpose of this repository is to share datasets describing the temporal complementarity of renewable energy sources. Temporal complementarity refers to synergies that result from multiple generating sources being out of sync and therefore tending to produce energy at different times. In cases where energy sources are out of sync, hybridization of those sources into a single plant can stabilize their combined energy output while contributing to additional economic benefits.

## Studies on temporal complementarity

These datasets describe multiple facets of the temporal complementarity of co-located hybrid renewable energy systems throughout the United States. Several metrics characterizing the complementarity of generation profiles are provided on an annual and monthly basis (for both hourly and daily aggregations). These generation profiles are underpinned by hourly resource data (e.g.., the WIND Toolkit and National Solar Radiation Database (NSRDB)) spanning the multi-year period 2007-2013. The data include complementarity results for greater than 1.76 million individual locations within the continental United States (CONUS). 

The data are intended to accompany two publications on the topic of temporal complementarity. Please refer to these publications for additional background, detailed methodology and discussion around temporal complementarity of renewables. 

1) Harrison-Atlas, Dylan, Caitlin Murphy, Anna Schleifer, and Nicholas Grue. "Temporal complementarity and value of wind-PV hybrid systems across the United States." *Renewable Energy* 201 (2022): 111-123, [doi:10.1016/j.renene.2022.10.060](https://doi.org/10.1016/j.renene.2022.10.060); 

2) Murphy, Caitlin, Harrison-Atlas, Dylan, Grue, Nicholas, Gevorgian, Vahan, Gallego-Calderon, Juan, Elliot, Shiloh and Mosier,Thomas. “A Resource Assessment for FlexPower”. NREL Technical Report.

## Accessing the data
Due to their large file sizes, datasets for *wind-pv* and *hydro* complementarity are managed using Git Large File Storage (LFS). Unless explicitly requested, Git will only initially download references to these files using text pointers when the repository is cloned. After installing Git LFS, users can download the actual CSV files using a “git lfs fetch” command. Git LFS can be downloaded (https://git-lfs.github.com/) or installed via a virtual environment (e.g., https://anaconda.org/conda-forge/git-lfs).

## Visualizing the data

### Wind-PV complementarity

A Jupyter Notebook ([wind_pv_data_exploration.ipynb](wind_pv_data_exploration.ipynb)) is provided to show how to read in and visualize the complementarity datasets for *wind-pv* hybrids using Python. There is a separate dataset for each metric specific to either hourly or daily time scales. Each record in the datasets represents an individual location within the contiguous United States. As shown in the notebook, locations are mapped using longitude and latitude coordinates. Multi-year mean values for each of the metrics are reported as are the monthly means taken across the 2007-2013 period of analysis.

### Hydro-PV/Wind complementarity

The notebooks for hydropower-based complementarity provide a description of the process used to compute the stability and pearson coefficients. In total, there are four notebooks that detail the process for different combinations of resource pairs and two temporal scales: daily (aggregated into annual complementarity) and monthly complementarity. More details of the hydropower-based complementarity analysis, including the file description, folder structure, and general considerations, are provided [here](./data/hydro/README.md).
