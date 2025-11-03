import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import re


def estimate_pitcher_fpts_std(row):
    summary_df = pd.read_csv("./output_data/summary.csv")
    try:
        print(summary_df.loc[row["PLAYERID"]]["std"])
        return float(summary_df.loc[row["PLAYERID"]]["std"])
        # return float(row["std"])
    except KeyError:
        if row["hits"] == 0:
            return 10.1
        var = (
            4.0 * row["strikeout"]
            + 1.0 * row["total_outs"]
            + 0.36 * row["hits"]
            + 0.36 * row["walk"]
            + 4.0 * row["er"]
            + 16.0 * row["win"] * (1 - row["win"])
        )
        return (var**0.5) * 1.4


def estimate_hitter_fpts_std(row):
    """
    Estimate the DraftKings fantasy points standard deviation for a hitter.
    Assumes Poisson-distributed counts for events and binary for stolen bases.
    """

    # Extract mean event counts per game
    single = row["SINGLE"]
    double = row["DOUBLE"]
    triple = row["TRIPLE"]
    home_run = row["HR"]
    walk = row["walk"]  # includes BB + HBP
    rbi = row["rbi"]
    runs = row["runs"]
    sb = row["sb"]

    # DraftKings scoring weights
    var = (
        9 * single  # 3^2
        + 25 * double  # 5^2
        + 64 * triple  # 8^2
        + 100 * home_run  # 10^2
        + 4 * walk  # 2^2
        + 4 * rbi  # 2^2
        + 4 * runs  # 2^2
        + 25 * sb * (1 - sb)  # 5^2 * p * (1 - p)
    )

    return np.sqrt(var)


def build_player_metadata(players_by_id):
    player_metadata = {}
    for pid, rows in players_by_id.items():
        if len(rows) == 2:
            real_pos = "/".join([str(rows[0].POS), str(rows[1].POS)])
        else:
            real_pos = str(rows[0].POS)

        player_metadata[pid] = (
            real_pos,
            str(rows[0].TEAM),
            float(rows[0].FPTS),
            float(rows[0].CEILING) if pd.notnull(rows[0].CEILING) else None,
            float(rows[0].FLOOR) if pd.notnull(rows[0].FLOOR) else None,
            float(rows[0].SALARY) if pd.notnull(rows[0].SALARY) else None,
            str(rows[0].OPP),
            str(rows[0].NAME),
        )
    return player_metadata


def build_simple_player_metadata(players_by_id):
    player_metadata = {}
    for pid, rows in players_by_id.items():
        if len(rows) == 2:
            real_pos = "/".join([str(rows[0].POS), str(rows[1].POS)])
        else:
            real_pos = str(rows[0].POS)

        player_metadata[pid] = (
            real_pos,
            str(rows[0].TEAM),
            float(rows[0].FPTS),
            float(rows[0].SALARY) if pd.notnull(rows[0].SALARY) else None,
            str(rows[0].NAME),
            str(rows[0].OPP),
        )
    return player_metadata


def sample_projections(player_metadata):
    projections = {}
    for pid, (pos, team, fpts, ceiling, floor, *_rest) in player_metadata.items():
        if ceiling is not None and floor is not None:
            stdev = (ceiling - floor) / 2
            projections[pid] = np.random.normal(loc=fpts, scale=stdev)
        else:
            projections[pid] = fpts
    return projections


def calculate_decay(player_metadata, tracker):
    decay = {}

    for pid, (pos, team, *_rest) in player_metadata.items():
        if pos == "P":
            decay[pid] = tracker.get_pitcher_penalty(pid)
        else:
            primary_decay = tracker.get_team_stack_penalty(team)
            decay[pid] = primary_decay

    return decay


def calculate_decay_vectorized(player_ids, teams, is_pitcher, tracker):
    # === Precompute all penalties ===

    # For pitchers
    pitcher_penalties = np.array(
        [tracker.get_pitcher_penalty(pid) for pid in player_ids]
    )

    # For hitters
    team_penalties = np.array([tracker.get_team_stack_penalty(team) for team in teams])

    # === Use np.where to select the right penalty ===
    decay_array = np.where(is_pitcher, pitcher_penalties, team_penalties)

    # === Return as {pid: decay} dictionary ===
    return dict(zip(player_ids, decay_array))


def get_stack_team_counts(selected_ids, player_metadata):
    teams = []
    for pid in selected_ids:
        data = player_metadata.get(pid)
        if data and len(data) >= 2:
            pos, team = data[0], data[1]
            if pos != "P":
                teams.append(team)
    return Counter(teams)


def extract_pitchers_and_hitters(selected_ids, player_metadata):
    """
    Returns two lists: pitchers and hitters from selected player IDs.
    """
    pitchers = []
    hitters = []
    for pid in selected_ids:
        player = player_metadata.get(pid)
        if player[0] == "P":
            pitchers.append([pid, player])
        else:
            hitters.append([pid, player])
    return pitchers, hitters


def choose_stack_shape(tracker, stack_targets):
    shapes = list(stack_targets.keys())
    weights = np.array(
        [
            stack_targets[shape] * tracker.get_stack_shape_penalty(shape)
            for shape in shapes
        ]
    )
    probabilities = weights / weights.sum()
    return np.random.choice(shapes, p=probabilities)


def extract_lineup_features(lineup, player_metadata):
    pitchers = []
    team_groups = {}
    player_teams = {}

    for pid in lineup:
        pos, team, *_ = player_metadata[pid]
        player_teams[pid] = team
        if pos == "P":
            pitchers.append(pid)
        else:
            team_groups.setdefault(team, []).append(pid)

    stacks = [group for group in team_groups.values() if len(group) >= 2]

    return {
        "pitchers": pitchers,
        "stacks": stacks,
        "player_teams": player_teams,
    }


def derive_pitcher_scores(
    game_sims: np.ndarray,
    team_to_hitters: Dict[str, List[str]],
    pitcher_metadata: Dict[str, Tuple],
    id_to_idx: Dict[str, int],
    default_scaling: float = -0.25,
) -> Tuple[List[str], np.ndarray]:
    """
    Simulates pitcher FPTS per sim by negatively correlating with opposing hitter totals,
    while anchoring around each pitcher's median and using their ceiling/floor for variance.

    Returns:
        - pitcher_ids: list in order of columns of output matrix
        - pitcher_sims: np.ndarray (num_sims, num_pitchers)
    """
    num_sims = game_sims.shape[0]
    pitcher_ids = [pid for pid, data in pitcher_metadata.items() if data[0] == "P"]
    pitcher_sims = np.zeros((num_sims, len(pitcher_ids)), dtype=np.float32)

    for i, pid in enumerate(pitcher_ids):
        pos, team, median, ceiling, floor, salary, opp_team, name = pitcher_metadata[
            pid
        ]
        if not opp_team or opp_team not in team_to_hitters:
            continue

        hitter_ids = [row.PLAYERID for row in team_to_hitters[opp_team]]
        hitter_indices = [id_to_idx[hid] for hid in hitter_ids if hid in id_to_idx]
        opp_totals = game_sims[:, hitter_indices].mean(axis=1)

        if ceiling is not None and floor is not None:
            stdev = abs(ceiling - floor) / 2
            print(stdev)
        else:
            stdev = 10.1

        pitcher_fpts = (
            median
            + default_scaling * opp_totals
            + np.random.normal(0, stdev, size=num_sims)
        )
        pitcher_sims[:, i] = pitcher_fpts
    print(pitcher_sims)
    return pitcher_ids, pitcher_sims


def get_id(row):
    # print(row)
    dk_raw = pd.read_csv(
        "../test_slates/large/DKSalaries.csv", skiprows=7, engine="python"
    )
    dk_df = dk_raw.iloc[:, 11:].copy()
    dk_df = dk_df.rename(columns=lambda c: c.strip())
    dk_df["Name"] = dk_df["Name"].str.strip()
    dk_df["Name"] = dk_df.apply(
        lambda row: re.sub(r"[^A-Z ]", "", row["Name"].strip()), axis=1
    )
    dk_df["Position"] = dk_df["Position"].str.strip()
    dk_df.loc[dk_df["Position"] == "SP", "Position"] = "P"
    dk_df.loc[dk_df["Position"] == "RP", "Position"] = "P"
    dk_df.loc[dk_df["TeamAbbrev"] == "CWS", "TeamAbbrev"] = "CHW"
    try:
        return dk_df[
            (dk_df["Name"] == re.sub(r"[^A-Z ]", "", row["NAME"].strip()))
            & (dk_df["TeamAbbrev"] == row["TEAM"])
            & (dk_df["Position"] == row["POS"])
            & (dk_df["Salary"] == row["SALARY"])
        ]["ID"].values[0]
    except IndexError:
        print(row)
        print(
            dk_df[
                (dk_df["Name"] == row["NAME"].strip())
                & (dk_df["TeamAbbrev"] == row["TEAM"])
            ]["ID"].values
        )
        raise
    except ValueError:
        print(row)


def cache_lineup_features(lineups, player_metadata):
    cache = {}

    for l in lineups:
        hitters = [pid for pid in l.player_ids if player_metadata[pid][0] != "P"]
        pitchers = sorted(
            [pid for pid in l.player_ids if player_metadata[pid][0] == "P"]
        )
        team_counts = defaultdict(int)
        for pid in hitters:
            team = player_metadata[pid][1]
            team_counts[team] += 1
        # Only consider stacks of size 3 or more
        candidates = [team for team, count in team_counts.items() if count >= 3]
        if not candidates:
            primary_stack = None
        else:
            # Break ties consistently using sorted team names
            primary_stack = sorted(candidates, key=lambda t: (-team_counts[t], t))[0]

        cache[frozenset(l.player_ids)] = {
            "pitchers": tuple(pitchers),
            "primary_stack": primary_stack,
        }

    return cache


def summarize_field_ownership(lineups, feature_cache):
    stack_counter = Counter()
    pitcher_counter = Counter()
    for l in lineups:
        features = feature_cache[frozenset(l.player_ids)]
        if features["primary_stack"] is not None:
            stack_counter[features["primary_stack"]] += 1
        pitcher_counter[features["pitchers"]] += 1
    return {
        "stack_ownership": stack_counter,
        "pitcher_ownership": pitcher_counter,
    }


def compute_field_distance(sampled_summary, projected_summary):
    def l1_norm(counter_a, counter_b):
        keys = set(counter_a.keys()) | set(counter_b.keys())
        total = 0
        for k in keys:
            a = counter_a.get(k, 0)
            b = counter_b.get(k, 0)
            total += abs(a - b)
        return total

    stack_diff = l1_norm(
        sampled_summary["stack_ownership"], projected_summary["stack_ownership"]
    )
    pitcher_diff = l1_norm(
        sampled_summary["pitcher_ownership"], projected_summary["pitcher_ownership"]
    )
    return stack_diff + pitcher_diff


def get_small_field_ownership_targets(pitcher_targets, stack_targets):
    p_target_mean = sum(pitcher_targets.values()) / len(pitcher_targets)
    s_target_mean = sum(stack_targets.values()) / len(stack_targets)
    for pid, target in pitcher_targets.items():
        if target > p_target_mean:
            constant = 1
        else:
            constant = -1
        delta = max(
            0,
            target
            + (constant * abs((target - p_target_mean) / (0.5 * len(pitcher_targets)))),
        )
        pitcher_targets[pid] = delta
    for team, target in stack_targets.items():
        if target > s_target_mean:
            constant = 1
        else:
            constant = -1
        stack_targets[team] = max(
            0,
            target
            + (constant * abs((target - s_target_mean) / (0.5 * len(stack_targets)))),
        )
    s_target_sum = sum(stack_targets.values())
    p_target_sum = sum(pitcher_targets.values())
    for pid, target in pitcher_targets.items():
        pitcher_targets[pid] = target / p_target_sum
    for team, target in stack_targets.items():
        stack_targets[team] = target / s_target_sum
    return pitcher_targets, stack_targets
