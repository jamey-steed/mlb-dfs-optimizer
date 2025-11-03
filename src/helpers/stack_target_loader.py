import pandas as pd
from .TEAM_NAME_TRANSLATION import TEAM_NAME_TRANSLATION


def load_stack_targets_combined(stack_csv_path, pitchers_df):
    df = pd.read_csv(stack_csv_path)
    df = df.rename(columns=lambda c: c.strip())
    df = df.rename(columns={"Opp. SP Own": "SPOwn%"})

    # TEAM STACK TARGETS
    team_stack_df = df[df["Ownership share"].notnull()]
    team_stack_df["Translated"] = (
        team_stack_df["Team"].map(TEAM_NAME_TRANSLATION).fillna(team_stack_df["Team"])
    )
    total = team_stack_df["Ownership share"].sum()
    team_targets = dict(
        zip(
            team_stack_df["Translated"].str.strip(),
            team_stack_df["Ownership share"] / total,
        )
    )

    # PITCHER OWNERSHIP TARGETS
    sp_own_df = df[df["SPOwn%"].notnull()]
    p_total = sp_own_df["SPOwn%"].sum()
    sp_own_df["TranslatedTeam"] = (
        sp_own_df["Team"].map(TEAM_NAME_TRANSLATION).fillna(sp_own_df["Team"])
    )
    team_to_spown = dict(zip(sp_own_df["TranslatedTeam"], sp_own_df["SPOwn%"]))

    valid_pitchers = pitchers_df[pd.notnull(pitchers_df["SALARY"])]
    pitcher_targets = {}

    for team, group in valid_pitchers.groupby("OPP"):
        if team == "CWS":
            if "CHW" in team_to_spown:
                top = group.loc[group["FPTS"].idxmax()]
                pitcher_targets[top["PLAYERID"].item()] = team_to_spown["CHW"] / p_total
        elif team in team_to_spown:
            top = group.loc[group["FPTS"].idxmax()]
            pitcher_targets[top["PLAYERID"].item()] = team_to_spown[team] / p_total
    return pitcher_targets, team_targets
