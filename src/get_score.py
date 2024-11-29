import os
import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse

def get_result(tmp):
    """
    Extracts the result from the given JSON object.
    
    Parameters:
    - tmp (dict): The JSON object containing the battle results.
    
    Returns:
    - dict: A dictionary containing the models and the winner, or None if no result is found.
    """
    obj = None
    try:            
        t = "Overall Judge"
        if "A" in tmp[t].strip():
            obj = {
                "model_a": tmp["model a"],
                "model_b": tmp["model b"],
                "win": "model_a",
            }
        if "B" in tmp[t].strip():
            obj = {
                "model_a": tmp["model a"],
                "model_b": tmp["model b"],
                "win": "model_b",
            }
        if "tie" in tmp[t].lower():
            obj = {
                "model_a": tmp["model a"],
                "model_b": tmp["model b"],
                "win": "tie",
            }
    except Exception as e:
        print(f"Error processing JSON: {e}")
    return obj

def compute_pairwise_win_fraction(battles):
    """
    Computes the pairwise win fraction for each model.
    
    Parameters:
    - battles (DataFrame): A DataFrame containing the battle results.
    
    Returns:
    - DataFrame: A DataFrame showing the win fraction for each model pair.
    """
    # Times each model wins as Model A
    a_win_ptbl = pd.pivot_table(
        battles[battles['win'] == "model_a"], 
        index="model_a", columns="model_b", aggfunc="size", fill_value=0)

    # Table counting times each model wins as Model B
    b_win_ptbl = pd.pivot_table(
        battles[battles['win'] == "model_b"], 
        index="model_a", columns="model_b", aggfunc="size", fill_value=0)

    # Table counting number of A-B pairs
    num_battles_ptbl = pd.pivot_table(battles, 
        index="model_a", columns="model_b", aggfunc="size", fill_value=0)

    # Computing the proportion of wins for each model as A and as B 
    # against all other models
    row_beats_col_freq = (
        (a_win_ptbl + b_win_ptbl.T) / 
        (num_battles_ptbl + num_battles_ptbl.T)
    )

    # Arrange ordering according to proportion of wins
    prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
    model_names = list(prop_wins.keys())
    row_beats_col = row_beats_col_freq.loc[model_names, model_names]
    return row_beats_col

def compute_bt(
    df, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None
):
    """
    Computes the Bradley-Terry model ratings for the given battles.
    
    Parameters:
    - df (DataFrame): A DataFrame containing the battle results.
    - SCALE (int): The scale factor for the ratings.
    - BASE (int): The base for the logarithmic transformation.
    - INIT_RATING (int): The initial rating for each model.
    - sample_weight (array-like): Optional sample weights.
    
    Returns:
    - Series: A Series containing the computed Elo ratings for each model.
    """
    from sklearn.linear_model import LogisticRegression
    
    ptbl_a_win = pd.pivot_table(
        df[df["win"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    
    # If no tie, create a zero matrix
    if sum(df["win"].isin(["tie", "tie (bothbad)"])) == 0:
        ptbl_tie = pd.DataFrame(0, index=ptbl_a_win.index, columns=ptbl_a_win.columns)
    else:
        ptbl_tie = pd.pivot_table(
            df[df["win"].isin(["tie", "tie (bothbad)"])],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
        ptbl_tie = ptbl_tie + ptbl_tie.T

    ptbl_b_win = pd.pivot_table(
        df[df["win"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    
    # Ensure all model pairs are accounted for
    all_models = set(ptbl_a_win.index).union(ptbl_a_win.columns)
    for model in all_models:
        if model not in ptbl_a_win.index:
            ptbl_a_win.loc[model] = 0
        if model not in ptbl_a_win.columns:
            ptbl_a_win[model] = 0
        if model not in ptbl_tie.index:
            ptbl_tie.loc[model] = 0
        if model not in ptbl_tie.columns:
            ptbl_tie[model] = 0
        if model not in ptbl_b_win.index:
            ptbl_b_win.loc[model] = 0
        if model not in ptbl_b_win.columns:
            ptbl_b_win[model] = 0

    # Reorder indices and columns
    ptbl_a_win = ptbl_a_win.reindex(index=all_models, columns=all_models, fill_value=0)
    ptbl_tie = ptbl_tie.reindex(index=all_models, columns=all_models, fill_value=0)
    ptbl_b_win = ptbl_b_win.reindex(index=all_models, columns=all_models, fill_value=0)

    # Combine win, loss, and tie counts
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            # Skip if NaN
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)

def pretty_print_model_ratings(ratings):
    """
    Pretty prints the model ratings in a DataFrame format.
    
    Parameters:
    - ratings (Series): A Series containing the Elo ratings for each model.
    
    Returns:
    - DataFrame: A DataFrame containing the model names and their corresponding Elo ratings.
    """
    df = pd.DataFrame([
        [n, ratings[n]] for n in ratings.keys()
    ], columns=["Model", "Elo rating"]).sort_values("Elo rating", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df

def main():
    parser = argparse.ArgumentParser(description="Compute model ratings based on battle results.")
    parser.add_argument("--input_dir", type=str, default="", help="Directory containing the input JSON files.")
    parser.add_argument("--history_path", type=str, help="Path to the history battle JSON file.")
    args = parser.parse_args()

    data = []
    model_answers = dict()

    if args.input_dir:
        print(f"Processing input directory: {args.input_dir}")
        for file in os.listdir(args.input_dir):
            file_path = os.path.join(args.input_dir, file)
            try:
                tmp = json.load(open(file_path, "r"))
                obj = get_result(tmp)
                if obj:
                    data.append(obj)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON file: {file_path}")
            except Exception as e:
                print(f"Unexpected error processing file {file_path}: {e}")

    if args.history_path:
        print(f"Loading history battles from: {args.history_path}")
        history_battles = json.load(open(args.history_path, "r"))
        for history_battle in history_battles:
            obj = get_result(history_battle)
            if obj:
                data.append(obj)

    battles = pd.DataFrame(data)

    print("\nPairwise Win Fractions:")
    print(compute_pairwise_win_fraction(battles).mean(axis=1).sort_values(ascending=False))

    print("\nComputing Bradley-Terry Ratings...")
    bt_ratings = compute_bt(battles)
    print("\nModel Ratings:")
    print(pretty_print_model_ratings(bt_ratings))

if __name__ == "__main__":
    main()