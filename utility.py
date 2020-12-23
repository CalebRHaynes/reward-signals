'''
Utility Script for running behavioral analyses - rutgers_socreward.ipynb
'''


import glob
import itertools
from scipy.io import loadmat
import numpy as np
import pandas as pd

# specify regular expression for all .mat behavioral files
file_list = glob.glob("data/10*/*feedback_[1-4].mat")

# specify all columns in raw data
col_list = [
    "Npoints",
    "lapse1",
    "lapse2",
    "deckchoice",
    "RT1",
    "RT2",
    "choice_onset",
    "press1_onset",
    "info_onset",
    "partner_onset",
    "press2_onset",
    "aff_onset",
    "fix1_list",
    "fix2_list",
    "fix3_list",
    "fix4_list",
    "soc_win",
    "block",
    "duration",
    "frequency",
    "stimaff",
    "stiminf",
    "point_total",
    "word",
    "rating",
    "is_catch",
    "partner",
    "highval_count",
]

def convert_mat(file):
    # read in raw behavioral .mat files -> pandas dataframes
    return pd.DataFrame(
        [[row.flat[0] for row in line] for line in loadmat(file)["data"][0]],
        columns=col_list)


def merge_dataframes(mat_file_list):
    # merge lists of .mat files into one pandas dataframe
    return pd.concat([convert_mat(file) for file in mat_file_list])


def clean_df(df):
    # eliminating rows deemed lapses
    return df.loc[
        ~(
            (df["lapse1"] == 1)
            | (df["lapse2"] == 1)
            | (df["deckchoice"] == 11)
            | (df["deckchoice"] == 22)
        )
    ].reset_index()


def add_known_val(df):

    """
    Adds columns for values participants could
    remember for the star and pentagon choices.

    Makes a point difference column that contains the
    difference of remembered values for current trial.
    """

    known_star = []
    known_pentagon = []
    for (index, row) in df.iterrows():

        # base case, participant starts with no info in lists
        if index == 0:
            known_star.append(np.nan)
            known_pentagon.append(np.nan)

        elif row.deckchoice == 1:  # star
            known_star.append(row.Npoints)  # star points
            # pentagon info is the same as previous
            known_pentagon.append(known_pentagon[-1])

        elif row.deckchoice == 2:  # pentagon
            known_pentagon.append(row.Npoints)  # pentagon points
            # star info is the same as previous
            known_star.append(known_star[-1])

    df["star_points"] = known_star  # star
    df["pentagon_points"] = known_pentagon  # pentagon

    # add point difference column
    diff_points = []
    for (star_val, pentagon_val) in zip(known_star, known_pentagon):
        # nan if these either values do not exist
        if star_val == np.nan or pentagon_val == np.nan:
            diff_points.append(np.nan)
        # get difference and add to list
        else:
            diff_points.append(pentagon_val - star_val)

    df["diff_points"] = diff_points

    return df


def add_prev_columns(df):
    """
    make previous trial partner column,
    affective feedback column
    """

    df["prev_is_social"] = df.partner.shift()
    df["aff_feedback_prev"] = df.soc_win.shift()
    df["prev_diff_points"] = df.diff_points.shift()

    return df

def col_name(df):

    # rename columns with dictionary
    renamed = {
        "index": "trial",
        "deckchoice": "choice",
        "partner": "is_social",
        "soc_win": "aff_feedback_curr"
    }

    return df.rename(columns = renamed)

#maybe include size of points in aff
#def split_aff_columns(df):

    #def label_aff_social (row):
    #    if row['prev_is_social'] == 1 :
    #        return row['aff_feedback_prev']
    #    else:
    #        return 0

    #def label_aff_nonsocial (row):
    #    if row['prev_is_social'] == 0 :
    #        return row['aff_feedback_prev']
    #    else:
    #        return 0

    #df['aff_feedback_prev_social'] = df.apply (lambda row: label_aff_social(row), axis=1)
    #df['aff_feedback_prev_nonsocial'] = df.apply (lambda row: label_aff_nonsocial(row), axis=1)

    #return df

def make_derived_df(matfiles, write=False):
    """
    Omnibus function for creating a derived dataframe
    to feed into regression.
    """

    # merge matfiles into a dataframe and eliminate non-essential rows
    df = clean_df(merge_dataframes(matfiles))

    # make known star, pentagon, and difference columns
    # add "trial - 1" (previous trial) columns
    df = add_prev_columns((add_known_val(df)))

    # rename columns and get rid of nans

    df = col_name(df)

    df = df.dropna()

    # optional logging as csv files
    if write == True:
        derived_df.to_csv(f"data_csv/sub-{subject[0][5:9]}.csv")

    return df
