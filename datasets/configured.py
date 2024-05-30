import os

import numpy as np
import pandas as pd

import config


def load_dfs_test_maize_us() -> tuple:
    path_data_us = os.path.join(config.PATH_DATA_DIR, "data_US")

    df_y = pd.read_csv(
        os.path.join(path_data_us, "county_data", "YIELD_COUNTY_US.csv"),
        index_col=["loc_id", "year"],
    )[["yield"]]

    df_x_soil = pd.read_csv(
        os.path.join(path_data_us, "county_data", "SOIL_COUNTY_US.csv"),
        index_col=["loc_id"],
    )[["sm_whc"]]

    df_x_meteo = pd.read_csv(
        os.path.join(path_data_us, "county_data", "METEO_COUNTY_US.csv"),
        index_col=["loc_id", "year", "dekad"],
    )

    df_x_rs = pd.read_csv(
        os.path.join(path_data_us, "county_data", "REMOTE_SENSING_COUNTY_US.csv"),
        index_col=["loc_id", "year", "dekad"],
    )

    dfs_x = (
        df_x_soil,
        df_x_meteo,
        df_x_rs,
    )

    df_y, dfs_x = _align_data(df_y, dfs_x)

    return df_y, dfs_x


def load_dfs_test_maize_fr() -> tuple:
    path_data_fr = os.path.join(config.PATH_DATA_DIR, "data_FR")

    df_y = pd.read_csv(
        os.path.join(path_data_fr, "YIELD_NUTS3_FR.csv"),
        index_col=["loc_id", "year"],
    )[["yield"]]

    df_x_soil = pd.read_csv(
        os.path.join(path_data_fr, "SOIL_NUTS3_FR.csv"),
        index_col=["loc_id"],
    )[["sm_whc"]]

    df_x_meteo = pd.read_csv(
        os.path.join(path_data_fr, "METEO_NUTS3_FR.csv"),
        index_col=["loc_id", "year", "dekad"],
    )

    df_x_rs = pd.read_csv(
        os.path.join(path_data_fr, "REMOTE_SENSING_NUTS3_FR.csv"),
        index_col=["loc_id", "year", "dekad"],
    )

    dfs_x = (
        df_x_soil,
        df_x_meteo,
        df_x_rs,
    )

    df_y, dfs_x = _align_data(df_y, dfs_x)

    return df_y, dfs_x


def load_dfs_test_maize() -> tuple:
    df_y_us, dfs_x_us = load_dfs_test_maize_us()
    df_y_fr, dfs_x_fr = load_dfs_test_maize_fr()

    df_y = pd.concat(
        [
            df_y_us,
            df_y_fr,
        ],
        axis=0,
    )

    dfs_x = tuple(
        pd.concat([df_x_us, df_x_fr], axis=0)
        for df_x_us, df_x_fr in zip(dfs_x_us, dfs_x_fr)
    )

    return df_y, dfs_x


def _align_data(df_y: pd.DataFrame, dfs_x: tuple) -> tuple:
    # Data Alignment
    # - Filter the label data based on presence within all feature data sets
    # - Filter feature data based on label data

    # Filter label data

    index_y_selection = set(df_y.index.values)
    for df_x in dfs_x:
        if len(df_x.index.names) == 1:
            index_y_selection = {
                (loc_id, year)
                for loc_id, year in index_y_selection
                if loc_id in df_x.index.values
            }

        if len(df_x.index.names) == 2:
            index_y_selection = index_y_selection.intersection(set(df_x.index.values))

        if len(df_x.index.names) == 3:
            index_y_selection = index_y_selection.intersection(
                set([(loc_id, year) for loc_id, year, _ in df_x.index.values])
            )

    # Filter the labels
    df_y = df_y.loc[list(index_y_selection)]

    # Filter feature data
    # TODO
    index_y_location_selection = set([loc_id for loc_id, _ in index_y_selection])

    return df_y, dfs_x


def load_dfs_test_softwheat_nl() -> tuple:
    path_data_nl = os.path.join(config.PATH_DATA_DIR, "data_NL")

    # TODO -- read crop calendar

    # Read crop calendar
    harvest_date = '12-31'  # TODO -- sensible dates
    sowing_date = '03-24'
    pre_sowing_timelength = 0  # TODO

    #
    # Soil
    #

    df_y = pd.read_csv(
        os.path.join(path_data_nl, "YIELD_NUTS2_NL.csv"),
        index_col=["loc_id", "year"],
    )

    df_y = df_y.loc[df_y["crop_name"] == "Soft wheat"][["yield"]]

    df_x_soil = pd.read_csv(
        os.path.join(path_data_nl, "SOIL_NUTS2_NL.csv"),
        index_col=["loc_id"],
    )[["sm_wp", "sm_fc", "sm_sat", "rooting_depth"]]

    #
    # Meteo
    #

    df_x_meteo = pd.read_csv(
        os.path.join(path_data_nl, "METEO_DAILY_NUTS2_NL.csv"),
    )

    df_x_meteo["date"] = pd.to_datetime(df_x_meteo["date"], format="%Y%m%d")
    df_x_meteo["year"] = df_x_meteo["date"].apply(lambda date: date.year)

    dfs_x_meteo = []
    for group, df_group in df_x_meteo.groupby(["loc_id", "year"]):
        loc_id, year = group

        # TODO -- obtain dates corresponding to crop and location from crop calendar

        df_group.set_index('date', inplace=True)
        df_group.resample('D').ffill()

        # TODO -- select data based on dates

        df_group.reset_index(inplace=True)
        dfs_x_meteo.append(df_group)

    df_x_meteo = pd.concat(dfs_x_meteo)
    df_x_meteo.set_index(["loc_id", "year", "date"], inplace=True)

    #
    # Remote sensing
    #

    df_x_rs = pd.read_csv(
        os.path.join(path_data_nl, "REMOTE_SENSING_NUTS2_NL.csv"),
    )

    df_x_rs["date"] = pd.to_datetime(df_x_rs["date"], format="%Y%m%d")
    df_x_rs["year"] = df_x_rs["date"].apply(lambda date: date.year)

    dfs_x_rs = []
    for group, df_group in df_x_rs.groupby(["loc_id", "year"]):
        loc_id, year = group

        # date_s = np.datetime64(f'{year}-{sowing_date}')
        # date_h = np.datetime64(f'{year}-{harvest_date}')  # TODO -- deal with sowing in previous year

        # season_length = date_h - date_s + np.timedelta64(pre_sowing_timelength, 'D')
        #
        # cutoff_date = date_h - season_length // 2

        df_group.set_index('date', inplace=True)
        df_group.resample('D').ffill()
        # df_group = df_group[cutoff_date:]

        df_group.reset_index(inplace=True)
        dfs_x_rs.append(df_group)

    df_x_rs = pd.concat(dfs_x_rs)

    df_x_rs.set_index(["loc_id", "year", "date"], inplace=True)

    dfs_x = (
        df_x_soil,
        df_x_meteo,
        df_x_rs,
    )

    df_y, dfs_x = _align_data(df_y, dfs_x)

    return df_y, dfs_x
