"""
AI Football Predictions Script: Predicts the likelihood of over 2.5 goals in upcoming football matches.

This script loads pre-trained machine learning models and match data to predict whether upcoming football matches will end with more than 2.5 goals. The predictions are then formatted into a Telegram-ready message that can be shared directly. 

How to run:
1. Ensure the necessary data and model files are in the specified directories.
2. Run the script with the appropriate arguments to generate predictions.
-----------------------------------------------------------------------------

Example usage, it is suggested to run the script in the root directory:

    python scripts/make_predictions.py --input_leagues_models_dir models --input_data_predict_dir data/processed --final_predictions_out_file data/final_predictions.json --next_matches data/next_matches.json

Required Libraries:
- pandas
- numpy
- pickle
- argparse
- datetime
- json
"""

from unittest import result

import pandas as pd
import os
import json
import pickle
import numpy as np
from datetime import datetime, timedelta
import argparse

# Define global constants
VALID_LEAGUES = ["E0", "I1", "D1", "SP1", "F1"]

# Define the features for home team, away team, and general match information
HOME_TEAM_FEATURES = [
    'HomeTeam', 'FTHG', 'HG', 'HTHG', 'HS', 'HST', 'HHW', 'HC', 'HF', 'HFKC', 'HO', 'HY', 'HR', 'HBP',
    'B365H', 'BFH', 'BSH', 'BWH', 'GBH', 'IWH', 'LBH', 'PSH', 'SOH', 'SBH', 'SJH', 'SYH', 'VCH', 'WHH',
    'BbMxH', 'BbAvH', 'MaxH', 'AvgH', 'BFEH', 'BbMxAHH', 'BbAvAHH', 'GBAHH', 'LBAHH', 'B365AHH', 'PAHH',
    'MaxAHH', 'AvgAHH', 'BbAHh', 'AHh', 'GBAH', 'LBAH', 'B365AH', 'AvgHomeGoalsScored', 'AvgHomeGoalsConceded',
    'HomeOver2.5Perc', 'AvgLast5HomeGoalsScored', 'AvgLast5HomeGoalsConceded', 'Last5HomeOver2.5Count', 'Last5HomeOver2.5Perc'
]

AWAY_TEAM_FEATURES = [
    'AwayTeam', 'FTAG', 'AG', 'HTAG', 'AS', 'AST', 'AHW', 'AC', 'AF', 'AFKC', 'AO', 'AY', 'AR', 'ABP',
    'B365A', 'BFA', 'BSA', 'BWA', 'GBA', 'IWA', 'LBA', 'PSA', 'SOA', 'SBA', 'SJA', 'SYA', 'VCA', 'WHA',
    'BbMxA', 'BbAvA', 'MaxA', 'AvgA', 'BFEA', 'BbMxAHA', 'BbAvAHA', 'GBAHA', 'LBAHA', 'B365AHA', 'PAHA',
    'MaxAHA', 'AvgAHA', 'AvgAwayGoalsScored', 'AvgAwayGoalsConceded', 'AwayOver2.5Perc', 'AvgLast5AwayGoalsScored',
    'AvgLast5AwayGoalsConceded', 'Last5AwayOver2.5Count', 'Last5AwayOver2.5Perc'
]

"""
The general features are common to both home and away teams and contain match information that is not specific to either team.
This list in no longer necessary because in case that a feature is not in the home or away team features, it will be considered as a general feature.
GENERAL_FEATURES = [
    'Div', 'Date', 'Time', 'FTR', 'Res', 'HTR', 'Attendance', 'Referee', 'Bb1X2', 'BbMxD', 'BbAvD', 'MaxD', 'AvgD',
    'B365D', 'BFD', 'BSD', 'BWD', 'GBD', 'IWD', 'LBD', 'PSD', 'SOD', 'SBD', 'SJD', 'SYD', 'VCD', 'WHD', 'BbOU',
    'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'GB>2.5', 'GB<2.5', 'B365>2.5', 'B365<2.5', 'P>2.5', 'P<2.5',
    'Max>2.5', 'Max<2.5', 'Avg>2.5', 'AvgC>2.5', 'Avg<2.5', 'AvgC<2.5', 'MaxCAHA', 'MaxC>2.5', 'B365C<2.5', 'MaxCA',
    'B365CAHH', 'BbAH', 'Over2.5'
]
"""

def load_model(filepath: str):
    """Loads the machine learning model from a specified pickle file.
    
    Args:
        filepath (str): Path to the pickle file containing the model.
    
    Returns:
        model: The loaded machine learning model.
    """
    try:
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {e}")


def load_league_data(filepath: str) -> pd.DataFrame:
    """Loads the league data from a CSV file using pandas.
    
    Args:
        filepath (str): Path to the CSV file containing league data.
    
    Returns:
        pd.DataFrame: The loaded league data as a DataFrame.
    """
    # check if the file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    else:
        print(f"Loading data from {filepath}...")
    # Load the data from the CSV file
        return pd.read_csv(filepath)


def prepare_row_to_predict(home_team_df: pd.DataFrame, away_team_df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
    """Prepares a DataFrame row for prediction by averaging relevant team statistics.
    
    Args:
        home_team_df (pd.DataFrame): DataFrame containing the home team's data.
        away_team_df (pd.DataFrame): DataFrame containing the away team's data.
        numeric_columns (list): List of numeric columns for prediction.
    
    Returns:
        pd.DataFrame: A single row DataFrame ready for prediction.
    """
    row_to_predict = pd.DataFrame(columns=numeric_columns)
    row_to_predict.loc[len(row_to_predict)] = [None] * len(row_to_predict.columns)

    home_team_final_df = home_team_df.head(5)[numeric_columns]
    away_team_final_df = away_team_df.head(5)[numeric_columns]

    for column in row_to_predict.columns:
        if column in HOME_TEAM_FEATURES:
            row_to_predict.loc[len(row_to_predict)-1, column] = home_team_final_df[column].mean()
        elif column in AWAY_TEAM_FEATURES:
            row_to_predict.loc[len(row_to_predict)-1, column] = away_team_final_df[column].mean()
        # If the column is not in the home or away team features, we take the average of both teams
        else:
            row_to_predict.loc[len(row_to_predict)-1, column] = (away_team_final_df[column].mean() + home_team_final_df[column].mean()) / 2

    return row_to_predict

def bucket_match(utc_iso: str) -> str:
    match_dt = datetime.strptime(utc_iso, '%Y-%m-%dT%H:%M:%SZ')
    today = datetime.utcnow().date()
    mdate = match_dt.date()

    if mdate == (today - timedelta(days=1)):
        return "yesterday"
    if mdate == today:
        return "today"
    return "upcoming"


def make_predictions_json(league: str, league_model, league_data: pd.DataFrame, competitions: dict) -> list:
    """
    Returns a list of match objects with predictions for a given league.
    """
    out = []

    league_info = competitions.get(league)
    if not league_info:
        return out

    for match in league_info["next_matches"]:
        home_team = match['home_team']
        away_team = match['away_team']

        # must exist in historical dataset
        if home_team not in league_data['HomeTeam'].values or away_team not in league_data['AwayTeam'].values:
            continue

        home_team_df = league_data[league_data['HomeTeam'] == home_team]
        away_team_df = league_data[league_data['AwayTeam'] == away_team]

        numeric_columns = league_data.select_dtypes(include=['number']).columns
        if 'Over2.5' in numeric_columns:
            numeric_columns = numeric_columns.drop('Over2.5')

        row_to_predict = prepare_row_to_predict(home_team_df, away_team_df, numeric_columns)
        X_test = row_to_predict.values

        pred = int(league_model.predict(X_test)[0])
        proba = league_model.predict_proba(X_test)[0]
        prob_under = float(proba[0])
        prob_over = float(proba[1])

        pick = "OVER" if pred == 1 else "UNDER"
        confidence = prob_over if pred == 1 else prob_under

        out.append({
            "matchId": match.get("matchId"),
            "league": league,
            "competitionName": league_info["name"],
            "competitionCrest": league_info["crest"],
            "utcDate": match.get("utcDate"),
            "date": match.get("date"),
            "status": match.get("status"),
            "matchday": match.get("matchday"),
            "bucket": bucket_match(match["utcDate"]) if match.get("utcDate") else "upcoming",
            "home_team": home_team,
            "away_team": away_team,
            "home_team_crest": match.get("home_team_crest"),
            "away_team_crest": match.get("away_team_crest"),
            "prediction": {
                "market": "over_2_5_goals",
                "pick": pick,
                "confidence": round(confidence, 6),
                "prob_over": round(prob_over, 6),
                "prob_under": round(prob_under, 6),
            }
        })

    return out

def main(input_leagues_models_dir: str, input_data_predict_dir: str, final_predictions_out_file: str, next_matches: str):
    try:
        print("Loading JSON file with upcoming matches...\n")
        # Use utf-8 (recommended) after you change acquire_next_matches to save utf-8
        with open(next_matches, 'r', encoding='utf-8') as json_file:
            competitions = json.load(json_file)
    except Exception as e:
        raise Exception(f"Error loading JSON file: {e}")

    result = {
        "generatedAt": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        "market": "over_2_5_goals",
        "competitions": {},
        "matches": []
    }

    for league in VALID_LEAGUES:
        print(f"----------------------------------")
        print(f"\nMaking predictions for {league}...\n")

        model_path = os.path.join(input_leagues_models_dir, f"{league}_voting_classifier.pkl")
        data_path = os.path.join(input_data_predict_dir, f"{league}_merged_preprocessed.csv")

        if not os.path.exists(model_path) or not os.path.exists(data_path):
            print(f"Missing data or model for {league}. Skipping...")
            continue

        league_model = load_model(model_path)
        league_data = load_league_data(data_path)

        league_matches = make_predictions_json(league, league_model, league_data, competitions)

        # store competition meta
        if league in competitions:
            result["competitions"][league] = {
                "name": competitions[league]["name"],
                "crest": competitions[league]["crest"]
            }

        result["matches"].extend(league_matches)

    # sort by utcDate
    result["matches"].sort(key=lambda m: m.get("utcDate") or "")

    # buckets for frontend
    result["buckets"] = {
        "yesterday": [m for m in result["matches"] if m["bucket"] == "yesterday"],
        "today": [m for m in result["matches"] if m["bucket"] == "today"],
        "upcoming": [m for m in result["matches"] if m["bucket"] == "upcoming"],
    }

    with open(final_predictions_out_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nPredictions JSON saved to {final_predictions_out_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Football Predictions Script")
    parser.add_argument('--input_leagues_models_dir', type=str, required=True, help="Directory containing the model files")
    parser.add_argument('--input_data_predict_dir', type=str, required=True, help="Directory containing the processed data files")
    parser.add_argument('--final_predictions_out_file', type=str, required=True, help="File path to save the Telegram message output")
    parser.add_argument('--next_matches', type=str, required=True, help="Path to the JSON file with upcoming matches information")

    args = parser.parse_args()
    main(args.input_leagues_models_dir, args.input_data_predict_dir, args.final_predictions_out_file, args.next_matches)
