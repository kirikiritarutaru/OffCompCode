
def wide_to_long(df_wide):
    return df_wide.reset_index().melt(id_vars='Date', var_name='User', value_name='Value').set_index('Date')

def long_to_wide(df_long):
    return df_long.reset_index().pivot(index='Date', columns='User', values='Value')