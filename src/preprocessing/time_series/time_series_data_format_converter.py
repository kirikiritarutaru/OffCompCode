import pandas as pd


def wide_to_long(df_wide):
    return df_wide.reset_index().melt(id_vars='Date', var_name='User', value_name='Value').set_index('Date')

def long_to_wide(df_long):
    return df_long.reset_index().pivot(index='Date', columns='User', values='Value')

if __name__ == '__main__':
    data_wide = {
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'User1': [10, 15, 20],
        'User2': [25, 30, 35]
    }
    df_wide = pd.DataFrame(data_wide)
    df_wide = df_wide.set_index('Date')

    # Example long format data
    data_long = {
        'Date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-03'],
        'User': ['User1', 'User2', 'User1', 'User2', 'User1', 'User2'],
        'Value': [10, 25, 15, 30, 20, 35]
    }
    df_long = pd.DataFrame(data_long)
    df_long = df_long.set_index('Date')

    print(df_wide)
    print(wide_to_long(df_wide))
    print(long_to_wide(wide_to_long(df_wide)))
