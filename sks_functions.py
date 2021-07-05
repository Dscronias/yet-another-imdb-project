#######################################################################
##################                   Programs
# Dimitri Scronias
##################
#######################################################################

# Dependencies
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW

# Functions
def fct_collapse(dict_to_explode: dict) -> dict:
    '''
    Explode values in dictionary.
    E.g. {"0-1": [0,1], "2-3": [2,3]} returns {0: "0-1", 1: "0-1", 2: "2-3", 3: "2-3}
    Use with df.replace(): df["bar"] = df["foo"].replace(ds_fct_collapse({"new_cat": ["oldcat1", "oldcat2",...]}))

    (Or use siuba)
    '''
    return dict(zip(
    pd.Series(dict_to_explode).explode().sort_values().values, 
    pd.Series(dict_to_explode).explode().sort_values().index)
    )

def gr_meanci(df, group, var):
    '''
    First argument: dataframe. Second argument: groupby. Third argument: quant var.
    Yields mean + CI of quant var by groupby
    '''
    return (pd.concat(
        [df.groupby(group).count()[var].rename("N"),
        df.groupby(group).mean()[var].rename("Mean"), 
        df.groupby(group).std()[var].rename("SD")], 
        axis=1)
        .assign(
            low_CI = lambda x: x.Mean - stats.t.ppf(0.975, x.N-1) * x.SD / np.sqrt(x.N),
            high_CI = lambda x: x.Mean + stats.t.ppf(0.975, x.N-1) * x.SD / np.sqrt(x.N))
        .drop(["SD"], axis=1)
    ).sort_values("Mean", ascending=False)    

def weighted_gr_meanci(df, group, var, w):
    '''
    First argument: dataframe. Second argument: groupby. Third argument: quant var.
    Yields mean + CI of quant var by groupby
    '''
    return df.groupby(group).agg(
        N=(var, "count"),
        Mean=(var, lambda x: DescrStatsW(x, weights=df.loc[x.index, w]/np.mean(df[w])).mean),
        LowCI=(var, lambda x: DescrStatsW(x, weights=df.loc[x.index, w]/np.mean(df[w])).tconfint_mean()[0]),
        HighCI=(var, lambda x: DescrStatsW(x, weights=df.loc[x.index, w]/np.mean(df[w])).tconfint_mean()[1])
    ) 

def tab(df, var, round_int):
    '''
    Yields N + % of a categorical variable.
    round_int = rounding decimal
    '''
    return pd.concat(
        [
            df[var].value_counts().rename("N"),
            round((df[var].value_counts()/len(df)*100), round_int).rename("%")
        ], axis=1
    )