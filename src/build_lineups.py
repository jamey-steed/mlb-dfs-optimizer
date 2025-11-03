import pandas as pd
import numpy as np
import time
from collections import Counter, defaultdict

from helpers.field_ownership import FieldOwnershipTracker
from helpers.stack_target_loader import load_stack_targets_combined
from build_test_lambda import build_from_lambda
from helpers.helper import (
    calculate_decay_vectorized,
    get_stack_team_counts,
    extract_pitchers_and_hitters,
    build_player_metadata,
    choose_stack_shape,
    get_id,
    estimate_pitcher_fpts_std,
    estimate_hitter_fpts_std,
    get_small_field_ownership_targets,
    sample_projections,
    calculate_decay,
)
from int_builder import (
    build_solver_5_3,
    build_solver_5_2,
    build_solver_5,
    build_solver_4_4,
    build_solver_4_3_1,
    build_solver_4,
    build_solver_3_3_1_1,
    build_solver_chaos,
)
import pickle

import cProfile
import pstats


# import io


def main(
    entry_fees: list[int],
    skip_build: bool = False,
    is_hybrid_build: bool = False,
    build_lambda: bool = False,
    start_from_prev_lineups_list: list[bool] = None,
    test_slate_size: str = "large",
):
    if start_from_prev_lineups_list is None:
        start_from_prev_lineups_list = [False] + [
            True for _ in range(len(entry_fees) - 1)
        ]
    # === Config ===
    SLATE_NAME = "Main"
    STACK_CSV_PATH = f"../test_slates/{test_slate_size}/stacks.csv"
    HITTERS_CSV_PATH = f"../test_slates/{test_slate_size}/hitters.csv"
    PITCHERS_CSV_PATH = f"../test_slates/{test_slate_size}/pitchers.csv"

    # === Load Data ===
    hitters = pd.read_csv(HITTERS_CSV_PATH)
    pitchers = pd.read_csv(PITCHERS_CSV_PATH)

    hitters = hitters.rename(
        columns={
            "single": "SINGLE",
            "double": "DOUBLE",
            "triple": "TRIPLE",
            "home_run": "HR",
            "pa": "PA",
            "Slate": "SLATE",
            "Position": "POS",
            "Team": "TEAM",
            "Opponent": "OPP",
            "Salary": "SALARY",
            "fpts": "FPTS",
            "Name": "NAME",
        }
    )
    pitchers = pitchers.rename(
        columns={
            "single": "SINGLE",
            "double": "DOUBLE",
            "triple": "TRIPLE",
            "home_run": "HR",
            "pa": "PA",
            "Slate": "SLATE",
            "Position": "POS",
            "Team": "TEAM",
            "Opponent": "OPP",
            "Salary": "SALARY",
            "fpts": "FPTS",
            "Name": "NAME",
            "Opp. SP Own": "SPOwn%",
        }
    )
    hitters.columns = [
        c.replace("/", "_").replace("%", "PCT").replace(" ", "_").replace("-", "_")
        for c in hitters.columns
    ]
    hitters = hitters[(hitters["SLATE"] == SLATE_NAME)]

    hitters["PLAYERID"] = hitters.apply(lambda row: get_id(row), axis=1)

    # Filter and clean pitchers
    pitchers = pitchers[
        (pitchers["SLATE"] == SLATE_NAME) & pd.notnull(pitchers["SALARY"])
    ]

    pitchers.columns = [
        c.replace("/", "_").replace("%", "PCT").replace(" ", "_").replace("-", "_")
        for c in pitchers.columns
    ]
    pitchers["POS"] = "P"

    pitchers["PLAYERID"] = pitchers.apply(lambda row: get_id(row), axis=1)
    pitchers["stdev"] = pitchers.apply(
        lambda row: estimate_pitcher_fpts_std(row), axis=1
    )
    pitchers["CEILING"] = pitchers["FPTS"] + pitchers["stdev"]
    pitchers["FLOOR"] = pitchers["FPTS"] - pitchers["stdev"]

    hitters["stdev"] = hitters.apply(lambda row: estimate_hitter_fpts_std(row), axis=1)
    hitters["CEILING"] = hitters["FPTS"] + hitters["stdev"]
    hitters["FLOOR"] = hitters["FPTS"] - hitters["stdev"]

    # Expand multi-position hitters
    expanded_hitters = []
    for _, row in hitters.iterrows():
        for pos in row["POS"].split("/"):
            new_row = row.copy()
            new_row["POS"] = pos
            expanded_hitters.append(new_row)
    hitters_df = pd.DataFrame(expanded_hitters)

    # Combine player pool
    all_players = pd.concat([pitchers, hitters_df], ignore_index=True)

    # === Preprocessing for fast access ===
    players_by_id = defaultdict(list)
    team_to_hitters = defaultdict(list)
    team_to_pitchers = defaultdict(list)
    team_to_opp = defaultdict(list)
    pos_to_players = defaultdict(list)
    for _, row in all_players.iterrows():
        for position in str(row.POS).split("/"):
            pos_to_players[position].append(row.PLAYERID)
        players_by_id[int(row.PLAYERID)].append(row)
        if row.TEAM not in team_to_opp:
            team_to_opp[row.TEAM] = row.OPP
        if row.POS == "P":
            team_to_pitchers[row.TEAM].append(row)
        else:
            team_to_hitters[row.TEAM].append(row)

    # === Load ownership targets ===
    pitcher_targets, team_targets = load_stack_targets_combined(
        STACK_CSV_PATH, pitchers
    )

    player_metadata = build_player_metadata(players_by_id)
    player_ids = list(player_metadata.keys())
    positions = np.array([player_metadata[pid][0] for pid in player_ids])
    teams = np.array([player_metadata[pid][1] for pid in player_ids])
    is_pitcher = positions == "P"
    for pitcher, target in pitcher_targets.items():
        print(player_metadata[pitcher][-1], target)
    print(team_targets)
    stack_targets = {
        "5-3": 0.18,
        "5-2": 0.26,
        "5-1-1": 0.15,
        "4-4": 0.08,
        "4-3-1": 0.1,
        "4-misc": 0.13,  # e.g., 4-2-1-1, 4-1-1-1-1
        "3-3-1-1": 0.02,
        "chaos": 0.08,  # might want to remove this at some point
    }
    for contest_num, entry_fee in enumerate(entry_fees):
        start_from_prev_lineups = start_from_prev_lineups_list[contest_num]
        field_sizes = {
            111: 1501,
            5: 47562,
            18: 9803,
            15: 7843,
            333: 1001,
            8: 33088,
            150: 259,
            190: 334,
            555: 300,
            222: 5005,
            55: 826,
            10: 23529,
            121: 550,
            250: 355,
            888: 600,
        }
        field_size = field_sizes[entry_fee]
        ENTRY_FEE = entry_fee
        LINEUPS_TO_SELECT = {
            111: 5,
            5: 150,
            18: 20,
            15: 10,
            333: 4,
            8: 50,
            150: 1,
            190: 1,
            555: 2,
            222: 10,
            55: 9,
            10: 150,
            121: 2,
            250: 4,
            888: 1,
        }[ENTRY_FEE]

        NUM_LINEUPS = (
            int(round(2 * field_size))
            if field_size < 20000
            else int(round(1.25 * field_size))
        )
        need_to_build = NUM_LINEUPS
        NUM_GENERATIONS = 30 if field_size < 20000 else 24
        if field_size < 2000:
            pitcher_targets, team_targets = get_small_field_ownership_targets(
                pitcher_targets, team_targets
            )

        tracker = FieldOwnershipTracker(
            total_lineups=NUM_LINEUPS,
            stack_shape_targets=stack_targets,
            team_stack_targets=team_targets,
            pitcher_targets={
                int(pitcher): target for pitcher, target in pitcher_targets.items()
            },
        )

        print(
            f"Starting section for {entry_fee} field size {field_size}, using prev lineups: {start_from_prev_lineups}"
        )

        if is_hybrid_build:
            with open("field_lineups.pkl", "rb") as f:
                raw_lineups = pickle.load(f)

            valid_ids = set(player_metadata.keys())
            salvaged = 0

            for pid_list in raw_lineups:
                if all(pid in valid_ids for pid in pid_list):
                    try:
                        team_counts = get_stack_team_counts(pid_list, player_metadata)

                        for team, count in team_counts.items():
                            if count >= 4:
                                tracker.update_primary_team(team)
                            elif count >= 2:
                                tracker.update_secondary_team(team)

                        shape = "-".join(
                            str(c) for c in sorted(team_counts.values(), reverse=True)
                        )
                        tracker.update_stack_shape(shape)

                        pitchers = [
                            pid for pid in pid_list if player_metadata[pid][0] == "P"
                        ]
                        tracker.update_pitchers(pitchers)

                        salvaged += 1
                    except:
                        continue

            print(
                f"Hybrid field rebuild: successfully tracked {salvaged} valid lineups."
            )
            need_to_build = NUM_LINEUPS - salvaged

        if build_lambda and not skip_build:
            solver_start_time = time.time()
            if NUM_LINEUPS < 10000 and start_from_prev_lineups is False:
                lineups, prev_lineups = build_from_lambda(
                    NUM_LINEUPS, pitcher_targets, team_targets, player_metadata, True
                )
            else:
                lineups = build_from_lambda(
                    NUM_LINEUPS, pitcher_targets, team_targets, player_metadata
                )
        elif not skip_build:
            solvers = {
                "5-3": build_solver_5_3(players_by_id, team_to_opp),
                "5-2": build_solver_5_2(players_by_id, team_to_opp),
                "5-1-1": build_solver_5(players_by_id, team_to_opp),
                "4-4": build_solver_4_4(players_by_id, team_to_opp),
                "4-3-1": build_solver_4_3_1(players_by_id, team_to_opp),
                "4-misc": build_solver_4(players_by_id, team_to_opp),
                "3-3-1-1": build_solver_3_3_1_1(players_by_id, team_to_opp),
                "chaos": build_solver_chaos(players_by_id, team_to_opp),
            }
            # === Solve Lineups ===
            solver_start_time = time.time()
            lineups = []
            projections_matrix = {}
            for pid, (
                pos,
                team,
                fpts,
                ceiling,
                floor,
                salary,
                opp,
                name,
            ) in player_metadata.items():
                if ceiling is not None and floor is not None:
                    stdev = (ceiling - floor) / 2
                    projections_matrix[pid] = np.random.normal(
                        loc=fpts, scale=stdev, size=NUM_LINEUPS
                    )
                else:
                    projections_matrix[pid] = np.full(NUM_LINEUPS, fpts)
            decay = calculate_decay(player_metadata, tracker)
            if not is_hybrid_build:
                need_to_build = NUM_LINEUPS
            actual_builder_start_time = time.time()
            for i in range(need_to_build):
                # if i == 2:
                #     builder.solver.EnableOutput()
                # else:
                #     builder.solver.SuppressOutput()
                projections = sample_projections(player_metadata)
                if (i / NUM_LINEUPS) > 0.1 or is_hybrid_build:
                    decay = calculate_decay_vectorized(
                        player_ids, teams, is_pitcher, tracker
                    )
                else:
                    decay = {}
                projections = {
                    pid: projections_matrix[pid][i] for pid in projections_matrix
                }
                shape = choose_stack_shape(tracker, stack_targets)
                builder = solvers[shape]
                builder.set_objective(i, projections_matrix, decay)
                selected_ids = builder.solve()
                if not selected_ids:
                    print(f"[DEBUG] Lineup {i + 1}: Solver failed to return a result.")
                    continue

                pitchers_df, hitters_df = extract_pitchers_and_hitters(
                    selected_ids, player_metadata
                )

                if len(pitchers_df) != 2 or len(hitters_df) != 8:
                    print(f"[DEBUG] Lineup {i + 1}: Invalid lineup length.")
                    continue

                team_counts = get_stack_team_counts(selected_ids, player_metadata)
                for team, count in team_counts.items():
                    if count >= 4:
                        tracker.update_primary_team(team)
                    elif count >= 2:
                        tracker.update_secondary_team(team)

                tracker.update_stack_shape(shape)
                tracker.update_pitchers([p[0] for p in pitchers_df])

                if i < 3:
                    print(f"\n\U0001f4e6 Lineup {i + 1}")
                    print("Pitchers:")
                    for p in pitchers_df:
                        print(f"{players_by_id[p[0]][0].NAME}", p)

                    print("Hitters:")
                    for h in hitters_df:
                        print(f"{players_by_id[h[0]][0].NAME}", h)
                if i % 100 == 0 and i > 0:
                    print(
                        f"{i} of {need_to_build} ({round(i * 100 / need_to_build, 2)}%) at: {datetime.now()}"
                    )
                    print(
                        f"Solver estimate: {int((need_to_build - i) * (time.time() - actual_builder_start_time) // (60 * i))}min {int(((need_to_build - i) * (time.time() - actual_builder_start_time) // i) % 60)}s"
                    )
                lineups.append(selected_ids)

            solver_end_time = time.time()
            print("\n\U0001f9ee Final Summary:")
            print("Stack Shape Usage:", tracker.shape_counter)
            print("Pitcher Usage (Top 5):")
            for pid, count in tracker.pitcher_counter.most_common(5):
                name = players_by_id.get(pid, [{}])[0].get("NAME", "UNKNOWN")
                print(f"  {name} (ID: {pid}) - {count} lineups")
            print(
                "Primary Stack Usage:",
                [
                    [
                        t[0],
                        f"{round(100 * t[1] / sum(tracker.team_stack_counter.values()), 2)}%",
                    ]
                    for t in tracker.team_stack_counter.most_common()
                ],
            )
            print(
                "Secondary Stack Usage:",
                [
                    [
                        t[0],
                        f"{round(100 * t[1] / sum(tracker.mini_stack_counter.values()), 2)}%",
                    ]
                    for t in tracker.mini_stack_counter.most_common()
                ],
            )
            if is_hybrid_build:
                lineups += raw_lineups
            with open("field_lineups.pkl", "wb") as f:
                pickle.dump(lineups, f)

            solver_elapsed = solver_end_time - solver_start_time

            print(
                f"Solver Done in {int(solver_elapsed // 60)}m {solver_elapsed % 60:.2f}s"
            )
        else:
            from pickle import load

            with open("field_lineups.pkl", "rb") as f:
                lineups = load(f)
        # === Run Evolutionary Optimization ===
        # from run_evolution import run_evolution

        # Load player IDs in correct row order
        # with open("id_map.pkl", "rb") as file:
        #     id_to_idx = pickle.load(file)
        #
        # # Load sim matrix
        # full_game_sims = np.load("player_sims.npy", allow_pickle=True).astype(
        #     np.float32
        # )
        #
        # # Call evolution using player_metadata directly
        # from lineup import Lineup
        #
        # used_lineups = []
        # stack_ownership = Counter()
        # pitcher_ownership = Counter()
        # used_lineups_set = set()
        # for l in lineups:
        #     if tuple(sorted(l)) not in used_lineups_set:
        #         used_lineups_set.add(tuple(sorted(l)))
        #         try:
        #             lineup = Lineup.from_raw_list(l, player_metadata)
        #             used_lineups.append(l)
        #             for pitcher in lineup.features["pitchers"]:
        #                 pitcher_ownership[player_metadata[pitcher][7]] += 1
        #             if lineup.features["stack_shape"][0] >= 4:
        #                 stack_ownership[lineup.features["stack_teams"][0]] += 1
        #             if lineup.features["stack_shape"][1] >= 4:
        #                 stack_ownership[lineup.features["stack_teams"][1]] += 1
        #         except KeyError:
        #             continue
        #
        # print(f"num deduped lineups = {len(used_lineups)}")
        # print(
        #     [
        #         (name, round(own / len(used_lineups), 2))
        #         for name, own in pitcher_ownership.most_common()
        #     ]
        # )
        # print(
        #     [
        #         (team, round(own / len(used_lineups), 2))
        #         for team, own in stack_ownership.most_common()
        #     ]
        # )
        # if NUM_LINEUPS >= 10000:
        #     prev_lineups = used_lineups
        # if start_from_prev_lineups:
        #     with open("last_lineups.pkl", "rb") as f:
        #         prev_lineups = pickle.load(f)
        # else:
        #     with open("last_lineups.pkl", "wb") as f:
        #         pickle.dump(used_lineups, f)
        # stack_ev, player_ev = run_evolution(
        #     field_lineups=used_lineups,
        #     entry_fee=ENTRY_FEE,
        #     lineups_to_select=LINEUPS_TO_SELECT,
        #     sims_matrix=full_game_sims.astype(np.float32),
        #     id_to_idx=id_to_idx,
        #     player_metadata=player_metadata,
        #     team_to_hitters=team_to_hitters,
        #     team_to_pitchers=team_to_pitchers,
        #     field_size=field_size,
        #     generations=5,
        #     pos_to_players=pos_to_players,
        #     enable_tracking=True,
        #     pitcher_targets=pitcher_targets,
        #     team_targets=team_targets,
        #     starting_lineups=prev_lineups,
        #     get_targets=True,
        # )
        # team_evs = {}
        # for size in [4, 5]:
        #     for team in stack_ev[size]:
        #         if team in team_evs:
        #             team_evs[team] = (team_evs[team] + stack_ev[size][team]) / 2
        #         else:
        #             team_evs[team] = stack_ev[size][team]
        #
        # for stack in team_targets:
        #     team_targets[stack] *= team_evs[stack] ** 2
        # total_p = 0
        # total_s = 0
        # # for pitcher in pitcher_targets:
        # #     total_p += pitcher_targets[pitcher]
        # for stack in team_targets:
        #     total_s += team_targets[stack]
        # # for pitcher in pitcher_targets:
        # #     pitcher_targets[pitcher] /= total_p
        # for stack in team_targets:
        #     team_targets[stack] /= total_s
        # print(team_targets)
        # print(pitcher_targets)
        # if not skip_build:
        #     lineups = build_from_lambda(
        #         NUM_LINEUPS, pitcher_targets, team_targets, player_metadata
        #     )
        #
        # evolved_lineups = run_evolution(
        #     field_lineups=used_lineups,
        #     entry_fee=ENTRY_FEE,
        #     lineups_to_select=LINEUPS_TO_SELECT,
        #     sims_matrix=full_game_sims.astype(np.float32),
        #     id_to_idx=id_to_idx,
        #     player_metadata=player_metadata,
        #     team_to_hitters=team_to_hitters,
        #     team_to_pitchers=team_to_pitchers,
        #     field_size=field_size,
        #     generations=NUM_GENERATIONS,
        #     pos_to_players=pos_to_players,
        #     enable_tracking=True,
        #     pitcher_targets=pitcher_targets,
        #     team_targets=team_targets,
        #     starting_lineups=prev_lineups,
        # )
        #
        # # Save output
        # el_df = pd.DataFrame(lineup.__repr__().split(";") for lineup in evolved_lineups)
        # el_df = pd.concat(
        #     [el_df[[0, 1, 2, 3]], el_df[4].str.split(",", expand=True)], axis=1
        # )
        # # print(el_df)
        # el_df.columns = [
        #     "Salary",
        #     "EV",
        #     "Shape",
        #     "Teams",
        #     "P",
        #     "P",
        #     "C",
        #     "1B",
        #     "2B",
        #     "3B",
        #     "SS",
        #     "OF",
        #     "OF",
        #     "OF",
        #     "ERR",
        # ]
        # el_df.sort_values(by="EV", ascending=False)
        # el_df["EV"] = el_df["EV"].astype(float).round(4)
        # try:
        #     pd.DataFrame(el_df).to_csv(f"evolved_lineups_{entry_fee}.csv", index=False)
        # except:
        #     pd.DataFrame(el_df).to_csv(f"evolved_lineups1_{entry_fee}.csv", index=False)
        #     print("EVOLVED_LINEUPS OPEN - SAVED TO evolved_lineups_1.csv")
        # print("Saved evolved lineups to evolved_lineups.csv")


if __name__ == "__main__":
    from datetime import datetime
    import importlib

    sims = importlib.import_module("game_sims")
    sims.main()
    print(f"Code started at {datetime.now()}")

    pr = cProfile.Profile()
    pr.enable()

    main(
        entry_fees=[333],
        skip_build=False,
        is_hybrid_build=False,
        build_lambda=False,
        start_from_prev_lineups_list=[False],
        test_slate_size="large",
    )
    pr.disable()
    ps = pstats.Stats(pr).sort_stats("cumtime")
    ps.print_stats(20)
