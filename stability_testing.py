import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def groupby_fuc(data):
    ''' group be function by day
    Inputs:
    - data: data framed to be grouped
    Output:
    - mix: df newly grouped by data dataframe
    '''
    groupby = []
    groupby.append(data.index.dayofyear)
    data = data.groupby(groupby)
    return(data)

def cf_mix_eq(solar_wind_df,hydro_df):

    ''' Returns a df of CF_mix variable from the paper linked in mian
    Inputs:
    - solar_wind_df: df of solar or wind data. Data's index should be a time series. Colums == sites
    - hydro_df: df of solar or wind data. Data's index should be a time series. Colums == sites
    Output:
    - mix: df denoting cf_mix calc
    '''
    mix = (solar_wind_df + hydro_df) / 2
    #print(mix)
    mix = groupby_fuc(mix)
    #print(mix.describe())

    return(mix)


def _format_grp_names(grp_names):
    """
    Format groupby index values

    Parameters
    ----------
    grp_names : list
        Group by index values, these correspond to each unique group in
        the groupby

    Returns
    -------
    out : ndarray
        2D array of grp index values properly formatted as strings
    """
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
                 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
                 11: 'Nov', 12: 'Dec'}

    # pylint: disable=unnecessary-lambda
    year = lambda s: "{}".format(s)
    month = lambda s: "{}".format(month_map[s])
    hour = lambda s: "{:02d}:00UTC".format(s)

    grp_names = np.array(grp_names).T
    if len(grp_names.shape) == 1:
        grp_names = np.expand_dims(grp_names, 0)

    out = []
    for grp_i in grp_names:  # pylint: disable=not-an-iterable
        grp_max = grp_i.max()
        if grp_max <= 12:
            out.append(list(map(month, grp_i)))
        elif grp_max <= 23:
            out.append(list(map(hour, grp_i)))
        else:
            out.append(list(map(year, grp_i)))
    return np.array(out).T


def _create_names(index, stats):
        """
        Generate statistics names

        Parameters
        ----------
        index : pandas.Index | pandas.MultiIndex
            Temporal index, either month, hour, or (month, hour)
        stats : list
            Statistics to be computed

        Returns
        -------
        columns_map : dict
            Dictionary of column names to use for each statistic
        columns : list
            Column names to use
        """
        column_names = _format_grp_names(index)

        columns_map = {}
        columns = []
        for s in stats:
            cols = {i: '{}_{}'.format('-'.join(n), s) for i, n
                    in zip(index, column_names)}
            columns_map[s] = cols
            columns.extend(list(cols.values()))

        return columns_map, columns


def stability_coefficient(mix, ref):
        """
        Compute average stability coefficient

        Parameters
        ----------
        mix : pandas.DataFrame
            DataFrame of mixed solar and wind time-series
        ref : pandas.DataFrame
            DataFrame of reference (solar or wind) time-series

        Returns
        -------
        stab : ndarray
            Vector of the average stability coefficient for all days in the
            provided time-series data. Averages are by site.
        """

        mix = mix.groupby(mix.index.dayofyear)
        
        mix_var = mix.apply(_daily_variability)
        #print(mix_var)
        ref = ref.groupby(ref.index.dayofyear)
        ref_var = ref.apply(_daily_variability)

        stab = 1 - ((mix_var / ref_var) * (ref.mean() / mix.mean()))

        mask = np.isfinite(stab)
        if not np.all(mask):
            stab[~mask] = np.nan

        return stab.mean().values.astype(np.float32)

def _daily_variability(doy):
        var = np.sqrt(np.sum((doy - doy.mean())**2))

        return var


def pearson_correlation(hydro_sites,hydro, wind_solar):
    ''' Return a dataframe with the discharge data
    Inputs:
    - hydro_sites: list of hydro sites
    - hydro: hydro df
    - wind_solar: wind or solar df
    Output:
    - writes out to a csv
    '''
    d={}
    for s in hydro_sites:
        hydro_array= np.array(hydro[s].values)
        #We need to update this to pull whatever the hydro companion is in wind and solar.
        #Like which one does hydro link to in wind and solar
        wind_solar_data_array = np.array(wind_solar[s].values)

        d[s]=pearsonr(hydro_array, wind_solar_data_array)

    pears_df =pd.DataFrame.from_dict(d, orient='index')

    pears_df=pears_df.rename(columns={0: 'pearsons_coeff',1: 'p-val'})
    pears_df= pears_df.reset_index().rename(columns={'index': 'site_id'})
    pears_df.to_csv('./output/pearson_stats.csv')


def main_stability(hydro_df, other_df, other_name='solar'):
    
    sites = hydro_df.columns.values
    #print(other_df)
    cf_mix = cf_mix_eq(other_df,hydro_df)

    #print(cf_mix.head())
    hydro_df =groupby_fuc(hydro_df)
    other_df =groupby_fuc(other_df)

    ref = hydro_df

    cols_map, _ =_create_names(list(cf_mix.groups),  ['stability'])

    out_stats = {}
    for grp_name, mix_grp in cf_mix:
        col = cols_map['stability'][grp_name]

        #print(mix_grp)
        ref_grp = ref.get_group(grp_name)
        #print(ref_grp)

        msg = ('mixed and reference data shapes do not match! {} != {}'
               .format(mix_grp.shape, ref_grp.shape))
        assert mix_grp.shape == ref_grp.shape, msg
        out_stats[col] = stability_coefficient(mix_grp, ref_grp)
    #out_stats = [pd.DataFrame(out_stats, index=sites, dtype=np.float32)]

    out_stats = pd.DataFrame(out_stats, index=sites, dtype=np.float32)
    
    # Preping dfs before return
    cf_mix_df = cf_mix.mean()
    out_stats = out_stats.T.reset_index(drop=True)

    return cf_mix_df, out_stats


if __name__ == "__main__":
    """
    Notes:
        Original Paper: https://iopscience.iop.org/article/10.1088/1748-9326/aad8f6/pdf
        NREL Code Base: https://nrel.github.io/reVX/_modules/reVX/hybrid_stats/hybrid_stats.html#HybridStabilityCoefficient
        This code was derived from NREL's reVX code base. 
        
    Before Run:
        Update: 'hydro_data' - this should be a time series depicting capacity factor in hour steps.
        Update: 'wind_solar_data' -this should be a time series depicting capacity factor in hour steps. 
        Update: means = zip(['hydro', 'UPDATE', 'reference', 'mixed'],[hydro_data, wind_solar_data, ref, cf_mix])
            update the above to either 'wind' or 'solar' 
        Update: output to reflect local environments. 
    """
    hydro_data = pd.read_csv('./Dam_PV_Wind_Profiles_SCPoints/cf_pv_profiles_2007_testing_2.csv') #stand in for hydro
    wind_solar_data = pd.read_csv('./Dam_PV_Wind_Profiles_SCPoints/cf_wind_profiles_2007_testing_2.csv')

    time_index = pd.date_range('1/1/2012 00:00', periods=8760, freq='1H')


    hydro_data.index = time_index
    wind_solar_data.index = time_index

    site_index = hydro_data.index

    sites = hydro_data.columns.values

    #hydro_data.to_csv('hydro.csv')
    #wind_solar_data.to_csv('solar.csv')

    #Juan this function needs to be update. See function notes
    pearson_correlation(sites, hydro_data, wind_solar_data)


    cf_mix = cf_mix_eq(hydro_data,wind_solar_data)

    hydro_data =groupby_fuc(hydro_data)
    wind_solar_data =groupby_fuc(wind_solar_data)

    ref = hydro_data

    cols_map, _ =_create_names(list(cf_mix.groups),  ['stability'])

    out_stats = {}
    for grp_name, mix_grp in cf_mix:
        col = cols_map['stability'][grp_name]

        ref_grp = ref.get_group(grp_name)

        msg = ('mixed and reference data shapes do not match! {} != {}'
               .format(mix_grp.shape, ref_grp.shape))
        assert mix_grp.shape == ref_grp.shape, msg
        out_stats[col] = stability_coefficient(mix_grp, ref_grp)
    out_stats = [pd.DataFrame(out_stats, index=sites, dtype=np.float32)]

    #This will need to be updated wind vs solar
    means = zip(['hydro', 'wind', 'reference', 'mixed'],[hydro_data, wind_solar_data, ref, cf_mix])


    for name, data in means:
        _, cols = _create_names(list(data.groups),[f'{name}_cf'])
        mean_data = data.aggregate(np.nanmean).T.astype(np.float32)
        mean_data.columns = cols
        out_stats.append(mean_data)

    out_stats =pd.concat(out_stats, axis=1)
    out_stats=out_stats.T


    out_stats.to_csv('./output/testing_out_stats.csv')


