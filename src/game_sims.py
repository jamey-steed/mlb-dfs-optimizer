import numpy as np
import pandas as pd
import random
from collections import defaultdict, Counter
from helpers.helper import get_id, build_simple_player_metadata
from tqdm import tqdm
import cProfile
import pstats
from helpers.dfs_explore import run_exploration

# ------------------ Configuration ------------------ #
SIM_COUNT = 1000
INNING_OUTS = 3
OUTCOME_PTS = {"walk": 2, "single": 3, "double": 5, "triple": 8, "home_run": 10}
DK_PITCH_PTS = {
    "OUT": 0.75,  # 3 × 0.75  = 2.25 per inning
    "K": 2.0,
    "HIT": -0.6,
    "WALK": -0.6,  # walk OR hit-by-pitch
    "ER": -2.0,
    "WIN": 4.0,
    "CG": 2.5,
    "CGSO": 2.5,
    "NOHIT": 5.0,
}

BLOWUP_ER_DEFAULT = 6
OUTCOME_STRINGS = np.array(["out", "walk", "single", "double", "triple", "home_run"])
OUTCOME_CODES = {name: i for i, name in enumerate(OUTCOME_STRINGS)}
PREPROCESSED_LINEUPS = None
PRE_SAMPLED_OUTCOMES = None
ID_TO_IDX = None


# Matchups from DKSalaries file
games_df = pd.read_csv(
    "../test_slates/large/DKSalaries.csv", skiprows=7, engine="python"
)
temp_matchups = [game.split()[0].split("@") for game in games_df["Game Info"].unique()]
rainouts = []
# rainouts = [["PHI", "CHW"]]
MATCHUPS = []
for matchup in temp_matchups:
    for i in range(2):
        if matchup[i] == "CWS":
            matchup[i] = "CHW"
    if matchup not in rainouts:
        MATCHUPS.append(matchup)

# ------------------ Game Simulation ------------------ #


def init_globals(pre_lineups, pre_outcomes, id_map, pitchers_df):
    """
    Called once, before the multiprocessing pool is spun up.
    """
    global PREPROCESSED_LINEUPS, PRE_SAMPLED_OUTCOMES, ID_TO_IDX, TEAM_STARTER
    PREPROCESSED_LINEUPS = pre_lineups
    PRE_SAMPLED_OUTCOMES = pre_outcomes
    ID_TO_IDX = id_map
    # PITCH_STAT_LOG = log_list
    TEAM_STARTER = {
        row.TEAM: row  # fast dict lookup per game
        for row in pitchers_df.itertuples(index=False)
    }


# ------------ Pitcher helpers (no NumPy) ---------------- #


def sample_bf(row):
    """Draw a truncated-normal BF cap."""
    return int(
        np.clip(np.random.normal(row.BF_MEAN, row.BF_SD), row.BF_MIN, row.BF_MAX)
    )


def dk_pitch(box, got_win: bool) -> float:
    pts = (
        box["outs"] * DK_PITCH_PTS["OUT"]
        + box["ks"] * DK_PITCH_PTS["K"]
        + box["hits"] * DK_PITCH_PTS["HIT"]
        + box["walks"] * DK_PITCH_PTS["WALK"]
        + box["er"] * DK_PITCH_PTS["ER"]
    )
    if got_win:
        pts += DK_PITCH_PTS["WIN"]
    return pts


def maybe_pull(box, row, side, home_score, away_score):
    """
    Decide BEFORE the next PA whether the starter leaves.
    Mutates box in-place; returns bool: starter still active?
    """
    if not box["active"]:
        return False
    if row is None:
        return False

    yank = (box["bf"] >= box["cap"]) or (box["er"] >= row.BLOWUP_ER)
    if yank:
        box["active"] = False
        box["qual"] = box["outs"] >= 15  # 5.0 IP?
        if box["qual"]:
            lead = home_score - away_score
            if (side == "home" and lead > 0) or (side == "away" and lead < 0):
                box["prov_win"] = True
        return False
    return True


def get_outcome(weights):
    outcomes = ["out", "walk", "single", "double", "triple", "home_run"]
    return random.choices(outcomes, weights=weights[:6], k=1)[0]


def run_sim(sim_idx):
    rng = random.Random(sim_idx)
    num_players = len(ID_TO_IDX)
    fpts = np.zeros(num_players)
    wins = Counter()

    for matchup_idx, ((away, home), (away_batting, home_batting)) in enumerate(
        PREPROCESSED_LINEUPS.items()
    ):
        team_base = matchup_idx * 2 * 9
        (
            (away_ids, away_pts, home_pid, home_ppts),
            (home_ids, home_pts, away_pid, away_ppts),
            winner,
        ) = simulate_matchup(
            away, home, away_batting, home_batting, sim_idx, team_base, rng
        )
        if winner == 1:
            wins[home] += 1
        elif winner == -1:
            wins[away] += 1
        # hitters
        for pid, pts in zip(away_ids, away_pts):
            if pid is not None:
                fpts[ID_TO_IDX[pid]] = pts
        for pid, pts in zip(home_ids, home_pts):
            if pid is not None:
                fpts[ID_TO_IDX[pid]] = pts

        # pitchers
        if home_pid is not None:
            fpts[ID_TO_IDX[home_pid]] = home_ppts
        if away_pid is not None:
            fpts[ID_TO_IDX[away_pid]] = away_ppts

    return fpts, wins


def sample_all_outcomes(weights_matrix, num_sims=50000, num_samples=6):
    W = weights_matrix.shape[0]

    cdf = np.cumsum(weights_matrix, axis=1)
    cdf /= cdf[:, -1][:, None]

    # Expand to (SIM_COUNT, W, 6)
    cdf = np.broadcast_to(cdf, (num_sims, W, 6))
    rand_vals = np.random.rand(num_sims, W, num_samples)

    idx = (rand_vals[..., None] < cdf[:, :, None, :]).argmax(axis=3)
    return idx.astype(np.int8)  # shape: (num_sims, W, num_samples)


def put_on_base(batter_idx, outcome, bases):
    scored_runners = []
    if outcome == "walk" or outcome == "hbp":
        if bases[0] is not None:
            if bases[1] is not None:
                if bases[2] is not None:
                    scored_runners = [bases[2]]
                    bases[2] = bases[1]
                bases[1] = bases[0]
            else:
                bases[1] = bases[0]
        bases[0] = batter_idx
    elif outcome == "single":
        if bases[2] is not None:
            scored_runners.append(bases[2])
        if bases[1] is not None:
            scored_runners.append(bases[1])
        bases[2] = bases[0]
        bases[1] = None
        bases[0] = batter_idx
    elif outcome == "double":
        for i in [2, 1]:
            if bases[i] is not None:
                scored_runners.append(bases[i])
                bases[i] = None
        if bases[0] is not None:
            scored_runners.append(bases[0])
            bases[0] = None
        bases[1] = batter_idx
    elif outcome == "triple":
        for i in range(3):
            if bases[i] is not None:
                scored_runners.append(bases[i])
                bases[i] = None
        bases[2] = batter_idx
    elif outcome == "home_run":
        for i in range(3):
            if bases[i] is not None:
                scored_runners.append(bases[i])
                bases[i] = None
        scored_runners.append(batter_idx)
    return scored_runners


def simulate_matchup(
    away, home, away_batting, home_batting, sim_idx, team_idx_base, rng
):
    # Check if all players are automatic outs (OUT=1, others=0)
    def is_all_outs(batting):
        if all(np.allclose(w, [1, 0, 0, 0, 0, 0, 0, 0, 0]) for _, w in batting):
            print(batting)
        return all(np.allclose(w, [1, 0, 0, 0, 0, 0, 0, 0, 0]) for _, w in batting)

    skip_away = is_all_outs(away_batting)
    skip_home = is_all_outs(home_batting)

    if skip_away and skip_home:
        print("Skipping game: both teams are all automatic outs.")
        return ([], np.array([])), ([], np.array([]))

        # ---- Pitcher set-up --------------------------------------------
    try:
        away_p = TEAM_STARTER[away]
    except KeyError:
        away_p = None
    try:
        home_p = TEAM_STARTER[home]
    except KeyError:
        home_p = None
    if away_p is not None:
        away_cap = sample_bf(away_p) + 2
    else:
        away_cap = 0
    if home_p is not None:
        home_cap = sample_bf(home_p) + 2
    else:
        home_cap = 0
    pitch_stat = {
        "away": dict(
            bf=0,
            outs=0,
            ks=0,
            hits=0,
            walks=0,
            er=0,
            cap=away_cap,
            active=True,
            qual=False,
            prov_win=False,
        ),
        "home": dict(
            bf=0,
            outs=0,
            ks=0,
            hits=0,
            walks=0,
            er=0,
            cap=home_cap,
            active=True,
            qual=False,
            prov_win=False,
        ),
    }

    away_points = np.zeros(len(away_batting))
    home_points = np.zeros(len(home_batting))

    away_score, home_score = 0, 0
    away_batter, home_batter = 0, 0
    half_inning = 1

    is_ph_home = [False for _ in range(9)]
    is_ph_away = [False for _ in range(9)]
    while True:
        is_home_half = half_inning % 2 == 0

        # End the game if away team has batted 9 times and home team is all automatic outs
        if half_inning >= 19 and skip_home:
            break

        # Skip half-inning if team has all automatic outs
        if (is_home_half and skip_home) or (not is_home_half and skip_away):
            half_inning += 1
            continue

        # Skip bottom of 9th or later if home team is already winning
        if half_inning >= 18 and is_home_half and home_score > away_score:
            break
        # print(half_inning, home_score, away_score)
        outs = 0
        if half_inning >= 19:
            if is_home_half:
                bases = [None, (home_batter - 1) % 9, None]
            else:
                bases = [None, (away_batter - 1) % 9, None]
        else:
            bases = [None, None, None]
        if half_inning > 18:
            break
        if home_batter // 9 >= 8 or away_batter // 9 >= 8:
            break
        while outs < INNING_OUTS:
            # ---------- starter pull check BEFORE this PA -------------
            pitch_side = "home" if not is_home_half else "away"
            box = pitch_stat[pitch_side]
            row = home_p if pitch_side == "home" else away_p
            starter_active = maybe_pull(box, row, pitch_side, home_score, away_score)
            scored = []
            if home_batter // 9 >= 8 or away_batter // 9 >= 8:
                break
            if is_home_half:
                pid, weights = home_batting[home_batter % 9]
                # outcome = get_outcome(weights)
                outcome = PRE_SAMPLED_OUTCOMES[
                    sim_idx, (team_idx_base + 9) + (home_batter % 9), home_batter // 9
                ]
                outcome = OUTCOME_STRINGS[outcome]
                if outcome == "out":
                    outs += 1
                else:
                    scored = put_on_base(home_batter % 9, outcome, bases)
                    if home_batter // 9 > weights[8] // 1:
                        if rng.choices(
                            [True, False], weights=[weights[7], 1 - weights[7]], k=1
                        ):
                            is_ph_home[home_batter % 9] = True
                    if not is_ph_home[home_batter % 9]:
                        home_points[home_batter % 9] += OUTCOME_PTS[outcome] + 2 * len(
                            scored
                        )
                    for s in scored:
                        if not is_ph_home[s]:
                            home_points[s] += 2
                    home_score += len(scored)
                    # ----- keep or revoke provisional wins ---------------
                    if pitch_stat["home"]["prov_win"] and home_score <= away_score:
                        pitch_stat["home"]["prov_win"] = False
                    if pitch_stat["away"]["prov_win"] and away_score <= home_score:
                        pitch_stat["away"]["prov_win"] = False

                    if (
                        outcome in ["walk", "single"]
                        and bases[1] is None
                        and not is_ph_home[home_batter % 9]
                    ):
                        sb_weights = [weights[6], 1 - weights[6]]
                        if rng.choices([True, False], weights=sb_weights, k=1)[0]:
                            bases[0], bases[1] = None, bases[0]
                            home_points[home_batter % 9] += 5
                # ---------- update pitcher stats from this PA -------------
                if starter_active:  # still the starter
                    tgt = box  # already the right dict
                else:
                    tgt = None  # bullpen stats not tracked

                if tgt is not None:
                    tgt["bf"] += 1
                    if outcome == "out":
                        tgt["outs"] += 1
                        if rng.random() < row.P_K:
                            tgt["ks"] += 1
                    elif outcome == "walk":
                        tgt["walks"] += 1
                    else:  # any hit
                        tgt["hits"] += 1
                    tgt["er"] += len(scored)
                if half_inning >= 18 and home_score > away_score:
                    break

                home_batter += 1
            else:
                pid, weights = away_batting[away_batter % 9]
                outcome = PRE_SAMPLED_OUTCOMES[
                    sim_idx, team_idx_base + (away_batter % 9), away_batter // 9
                ]
                outcome = OUTCOME_STRINGS[outcome]
                if outcome == "out":
                    outs += 1
                else:
                    scored = put_on_base(away_batter % 9, outcome, bases)
                    if away_batter // 9 > weights[8] // 1:
                        if rng.choices(
                            [True, False], weights=[weights[7], 1 - weights[7]], k=1
                        ):
                            is_ph_away[away_batter % 9] = True
                    if not is_ph_away[away_batter % 9]:
                        away_points[away_batter % 9] += OUTCOME_PTS[outcome] + 2 * len(
                            scored
                        )
                    for s in scored:
                        if not is_ph_away[s]:
                            away_points[s] += 2
                    away_score += len(scored)
                    # ----- keep or revoke provisional wins ---------------
                    if pitch_stat["home"]["prov_win"] and home_score <= away_score:
                        pitch_stat["home"]["prov_win"] = False
                    if pitch_stat["away"]["prov_win"] and away_score <= home_score:
                        pitch_stat["away"]["prov_win"] = False

                    if (
                        outcome in ["walk", "single"]
                        and bases[1] is None
                        and not is_ph_away[away_batter % 9]
                    ):
                        sb_weights = [weights[6], 1 - weights[6]]
                        if rng.choices([True, False], weights=sb_weights, k=1)[0]:
                            bases[0], bases[1] = None, bases[0]
                            away_points[away_batter % 9] += 5
                # ---------- update pitcher stats from this PA -------------
                if starter_active:  # still the starter
                    tgt = box  # already the right dict
                else:
                    tgt = None  # bullpen stats not tracked

                if tgt is not None:
                    tgt["bf"] += 1
                    if outcome == "out":
                        tgt["outs"] += 1
                        if rng.random() < row.P_K:
                            tgt["ks"] += 1
                    elif outcome == "walk":
                        tgt["walks"] += 1
                    else:  # any hit
                        tgt["hits"] += 1
                    tgt["er"] += len(scored)

                away_batter += 1

        if half_inning >= 18 and is_home_half:
            if home_score > away_score:
                break  # walk-off
            elif outs >= INNING_OUTS and home_score < away_score:
                break  # home team failed to tie or win
        half_inning += 1

    away_ids = [row[0] for row in away_batting]
    home_ids = [row[0] for row in home_batting]

    away_ppts = dk_pitch(pitch_stat["away"], pitch_stat["away"]["prov_win"])
    home_ppts = dk_pitch(pitch_stat["home"], pitch_stat["home"]["prov_win"])
    # print(pitch_stat)
    if home_p is not None:
        home_pid = home_p.PLAYERID
    else:
        home_pid = None
    if away_p is not None:
        away_pid = away_p.PLAYERID
    else:
        away_pid = None
    if home_score > away_score:
        winner = 1
    elif home_score < away_score:
        winner = -1
    else:
        winner = 0
    # --- DEBUG LOG -------------------------------------------------------
    # Cast all numpy ints to plain Python ints so Manager().list() stays happy.
    # def clean(box):
    #     return {k: int(v) if isinstance(v, (np.integer, np.int_)) else v
    #             for k, v in box.items()}
    #
    # PITCH_STAT_LOG.append(dict(
    #     sim_idx=sim_idx,
    #     away_team=away,
    #     home_team=home,
    #     home_box=clean(pitch_stat["home"]),
    #     away_box=clean(pitch_stat["away"]),
    # ))

    return (
        (away_ids, away_points, home_pid, home_ppts),
        (home_ids, home_points, away_pid, away_ppts),
        winner,
    )


# ------------------ Placeholder Data Loaders ------------------ #


def load_hitters(test_slate_size: str = "large"):
    return pd.read_csv(f"../test_slates/{test_slate_size}/hitters.csv")


# ------------------ Pitcher Data ------------------ #
def load_pitchers(path: str = "../test_slates/large/pitchers.csv") -> pd.DataFrame:
    p = pd.read_csv(path)

    p.columns = [
        c.replace("/", "_").replace("%", "PCT").replace(" ", "_").replace("-", "_")
        for c in p.columns
    ]
    p.columns = [c.upper() for c in p.columns]
    p = p[pd.notnull(p["SALARY"])]
    p["POS"] = "P"
    # ------------ workload distribution -----------------------------------
    p["BF_MEAN"] = p["TBF"]  # projected batters faced
    p["OUTS_MEAN"] = p["TOTAL_OUTS"]  # projected outs
    p["BF_SD"] = 3  # ≈ half an inning of variance
    p["BF_MIN"] = 12  # at least through the order once
    p["BF_MAX"] = 36  # never more than 4× around

    # ------------ per-plate-appearance probabilities ----------------------
    p["P_K"] = p["STRIKEOUT"] / p["TOTAL_OUTS"].clip(lower=1)
    p["P_BB"] = p["WALK"] / p["TBF"].clip(lower=1)  # treat HBP as BB
    p["P_H"] = p["HITS"] / p["TBF"].clip(lower=1)  # includes HR
    # p["ER_RATE"] = p["ER"] / p["TBF"].clip(lower=1)  # unused now but handy

    # ------------ blow-up hook threshold ----------------------------------
    p["BLOWUP_ER"] = 6  # yank after 6 earned runs (can tune later)
    p["PLAYERID"] = p.apply(lambda row: get_id(row), axis=1)
    # print(p)
    return p


def find_substitute_weights(batting_order, i):
    for j in range(i + 1, 9):
        if batting_order[j] is not None:
            return batting_order[j][1].copy()
    for j in range(i - 1, -1, -1):
        if batting_order[j] is not None:
            return batting_order[j][1].copy()
    return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])


def pad_lineup(team_df):
    batting_order = [None] * 9
    for row in team_df.itertuples(index=False):
        # print(row.BattingOrder)
        idx = max(0, min(8, int(row.BattingOrder) - 1))
        # print(idx)
        batting_order[idx] = (
            row.PLAYERID,
            np.array(
                [
                    row.OUT_RATE,
                    row.BB_RATE,
                    row.SINGLE_RATE,
                    row.DOUBLE_RATE,
                    row.TRIPLE_RATE,
                    row.HR_RATE,
                    row.SB_RATE,
                    row.PH,
                    row.PA,
                ]
            ),
        )
    for i in range(9):
        if batting_order[i] is None:
            batting_order[i] = (None, find_substitute_weights(batting_order, i))
        elif batting_order[i][1][0] >= 1 or batting_order[i][1][0] <= 0:
            batting_order[i] = (
                batting_order[i][0],
                find_substitute_weights(batting_order, i),
            )
    # print(batting_order)
    return batting_order


def run_sim_batch(indices):
    fpts_batch = np.zeros((len(indices), len(ID_TO_IDX)), dtype=np.float32)
    wins_counter = Counter()
    for j, i in enumerate(indices):
        fpts, winners = run_sim(i)
        fpts_batch[j] = fpts
        wins_counter.update(winners)

    return np.stack(fpts_batch), wins_counter


def main():
    import pprint

    pp = pprint.PrettyPrinter(indent=2)
    hitters = load_hitters()
    pitchers_df = load_pitchers()
    hitters = hitters.rename(
        columns={
            "single": "SINGLE",
            "double": "DOUBLE",
            "triple": "TRIPLE",
            "home_run": "HR",
            "pa": "PA",
            "Name": "NAME",
            "Position": "POS",
            "Team": "TEAM",
            "Salary": "SALARY",
            "sb": "SB",
            "pinch_hit": "PH",
            "fpts": "FPTS",
            "Opponent": "OPP",
        }
    )
    pitchers_df = pitchers_df.rename(columns={"Opponent": "OPP"})
    hitters.columns = [
        c.replace("/", "_").replace("%", "PCT").replace(" ", "_").replace("-", "_")
        for c in hitters.columns
    ]
    hitters["PLAYERID"] = hitters.apply(lambda row: get_id(row), axis=1)
    hitters["BB_RATE"] = hitters["walk"] / hitters["PA"]
    hitters["SINGLE_RATE"] = hitters["SINGLE"] / hitters["PA"]
    hitters["DOUBLE_RATE"] = hitters["DOUBLE"] / hitters["PA"]
    hitters["TRIPLE_RATE"] = hitters["TRIPLE"] / hitters["PA"]
    hitters["HR_RATE"] = hitters["HR"] / hitters["PA"]
    hitters["OUT_RATE"] = 1 - (
        hitters["BB_RATE"]
        + hitters["SINGLE_RATE"]
        + hitters["DOUBLE_RATE"]
        + hitters["TRIPLE_RATE"]
        + hitters["HR_RATE"]
    )
    hitters["SB_RATE"] = hitters["SB"] / (hitters["walk"] + hitters["SINGLE"])
    preprocessed_lineups = {}
    for away, home in MATCHUPS:
        away_team_df = hitters[(hitters["TEAM"] == away)].sort_values("BattingOrder")
        home_team_df = hitters[(hitters["TEAM"] == home)].sort_values("BattingOrder")
        preprocessed_lineups[(away, home)] = (
            pad_lineup(away_team_df),
            pad_lineup(home_team_df),
        )

    unique_ids = np.concatenate(
        (hitters["PLAYERID"].unique(), pitchers_df["PLAYERID"].unique())
    )
    id_to_idx = {pid: i for i, pid in enumerate(unique_ids)}
    N_PLAYERS = len(unique_ids)
    player_fpts = np.zeros((SIM_COUNT, N_PLAYERS))
    # Flatten all batting orders and build mapping to each team's index
    all_teams = []
    team_batters = []

    for (away, home), (away_batting, home_batting) in preprocessed_lineups.items():
        all_teams.append((away, home))
        team_batters.append(away_batting)
        team_batters.append(home_batting)

    flat_weights = np.array(
        [weights[:6] for team in team_batters for _, weights in team]
    )  # shape: (T * 9, 6)

    # Pre-sample all outcomes for every player across all simulations
    PA_PER_HITTER = 8
    pre_sampled_outcomes = sample_all_outcomes(
        flat_weights, num_sims=SIM_COUNT, num_samples=PA_PER_HITTER
    )
    team_idx = 0
    # for sim in range(SIM_COUNT):  # Begin simulation loop
    #     if sim % 1000 == 0:
    #         print(f"{sim}/{SIM_COUNT}")
    #     for matchup_idx, (away, home) in enumerate(MATCHUPS):
    #         # print(f"{away}: {away_batting}")
    #         # print(f"{home}: {home_batting}")
    #         away_batting, home_batting = preprocessed_lineups[(away, home)]
    #         team_base = matchup_idx * 2 * 9
    #         result = simulate_matchup(away_batting, home_batting, sim, pre_sampled_outcomes, team_base)
    #         if result is None:
    #             continue
    #         (away_ids, away_pts), (home_ids, home_pts) = result
    #         for pid, pts in zip(away_ids, away_pts):
    #             if pid is not None:
    #                 player_fpts[sim, id_to_idx[pid]] = pts
    #         for pid, pts in zip(home_ids, home_pts):
    #             if pid is not None:
    #                 player_fpts[sim, id_to_idx[pid]] = pts
    print("running sims in parallel")
    from concurrent.futures import ProcessPoolExecutor
    from math import ceil

    N_PROCS = 7
    CHUNK_SIZE = ceil(SIM_COUNT / (N_PROCS))

    batches = [
        range(i, min(i + CHUNK_SIZE, SIM_COUNT))
        for i in range(0, SIM_COUNT, CHUNK_SIZE)
    ]

    results = []
    global_win_counter = Counter()
    with ProcessPoolExecutor(
        max_workers=N_PROCS,
        initializer=init_globals,
        initargs=(preprocessed_lineups, pre_sampled_outcomes, id_to_idx, pitchers_df),
    ) as executor:
        for fpts_chunk, batch_counter in tqdm(
            executor.map(run_sim_batch, batches), total=len(batches)
        ):
            results.append(fpts_chunk)
            global_win_counter.update(batch_counter)

    player_fpts = np.vstack(results)
    for away, home in MATCHUPS:
        total_games = global_win_counter[away] + global_win_counter[home]
        print(f"{away} win %: {round(global_win_counter[away] / total_games, 4) * 100}")
        print(f"{home} win %: {round(global_win_counter[home]/total_games,4)*100}")

    np.save("output_data/player_sims.npy", player_fpts.astype(np.float32))
    import pickle

    with open("output_data/id_map.pkl", "wb") as file:
        pickle.dump(id_to_idx, file)

    # Stack dominance calculation
    stack_dominance = Counter()
    mini = Counter()
    team_to_players = defaultdict(list)

    all_players = pd.concat([pitchers_df, hitters], ignore_index=True)

    # === Preprocessing for fast access ===
    players_by_id = defaultdict(list)
    for _, row in all_players.iterrows():
        players_by_id[int(row.PLAYERID)].append(row)
    player_metadata = build_simple_player_metadata(players_by_id)
    print(run_exploration(player_fpts, id_to_idx, player_metadata))

    for pid, idx in id_to_idx.items():
        try:
            team = hitters.loc[hitters["PLAYERID"] == pid, "TEAM"].values[0]
            team_to_players[team].append((pid, idx))
        except IndexError:
            pass

    for sim in range(SIM_COUNT):
        best_stacks = {}
        mini_stacks = {}
        for team, players in team_to_players.items():
            fpts_list = [(player_fpts[sim, idx], pid) for pid, idx in players]
            top5 = sorted(fpts_list, reverse=True)[:5]
            stack_score = sum([f for f, _ in top5])
            best_stacks[team] = stack_score
            top3 = sorted(fpts_list, reverse=True)[:3]
            mini_stacks[team] = sum([f for f, _ in top3])

        top_team = max(best_stacks.items(), key=lambda x: x[1])[0]
        stack_dominance[top_team] += 1
        top_mini = max(mini_stacks.items(), key=lambda x: x[1])[0]
        mini[top_mini] += 1

    print("Stack dominance (% of simulations where a team had top 5 stack score):")
    for team, count in sorted(
        stack_dominance.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{team} {100 * count / SIM_COUNT:.2f}")
    print("\n\n\n_________________MINI STACKS__________________")
    for team, count in sorted(mini.items(), key=lambda x: x[1], reverse=True):
        print(f"{team} {100 * count / SIM_COUNT:.2f}")

    nonempty_sims = np.count_nonzero(player_fpts.sum(axis=1))
    print(f"Simulations complete. Player FPTS saved to player_sims.npy")
    print(f"Actual nonempty simulations run: {nonempty_sims} of {SIM_COUNT}")
    # import json, gzip, datetime as dt
    # timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    # out_path = f"pitch_stat_log_{timestamp}.json.gz"
    #
    # with gzip.open(out_path, "wt") as f:
    #     json.dump(list(pitch_stat_log), f)

    # print(f"wrote {len(pitch_stat_log):,} games to {out_path}")

    summary_df = pitcher_fpt_summary(
        fpts=player_fpts, pitchers_df=pitchers_df, id_to_idx=id_to_idx, top_n=None
    )
    # print(summary_df)
    summary_df.to_csv("./output_data/summary.csv")


def pitcher_fpt_summary(
    fpts: np.ndarray, pitchers_df: pd.DataFrame, id_to_idx: dict, top_n: int = 20
) -> pd.DataFrame:
    """
    Returns a DataFrame of mean / std / min / max DK points
    for every starter in the slate (top_n by mean by default)
    and also produces an easy-to-read error-bar plot.
    """
    import pandas as pd

    # import numpy as np
    # import matplotlib.pyplot as plt
    # --- build summary table -------------------------------------------------
    rows = []
    for row in pitchers_df.itertuples(index=False):
        col = id_to_idx[row.PLAYERID]  # column where this pitcher lives
        pts = fpts[:, col]  # all simulated scores

        rows.append(
            dict(
                playerid=row.PLAYERID,
                name=f"{row.NAME} ({row.TEAM})",
                mean=pts.mean(),
                std=pts.std(ddof=0),
                fpts_min=pts.min(),
                fpts_max=pts.max(),
                sims=len(pts),
            )
        )
    pd.set_option("display.max_columns", None)
    df = pd.DataFrame(rows).set_index("playerid").sort_values("mean", ascending=False)
    if top_n is not None:
        df = df.head(top_n).reset_index(drop=True)

    # --- quick visual --------------------------------------------------------
    # plt.figure(figsize=(12, 6))
    # plt.errorbar(
    #     x=np.arange(len(df)),
    #     y=df["mean"],
    #     yerr=df["std"],
    #     fmt="o",
    #     ecolor="black",
    #     capsize=4,
    #     linewidth=1.2)
    # plt.xticks(np.arange(len(df)), df["name"], rotation=60, ha="right", fontsize=9)
    # plt.ylabel("DraftKings FPTS")
    # plt.title(f"Pitcher Sim Distribution – {df['sims'][0]:,} sims each\n"
    #           "(dots = mean, bars = ±1 σ; caps = min/max clipped)")
    # # extend whiskers to min/max for quick context
    # for i, (_, r) in enumerate(df.iterrows()):
    #     plt.plot([i, i], [r.fpts_min, r.fpts_max], lw=0.8, color="gray")
    # plt.tight_layout()
    # plt.show()

    return df


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    ps = pstats.Stats(pr).sort_stats("cumtime")
    ps.print_stats(20)
