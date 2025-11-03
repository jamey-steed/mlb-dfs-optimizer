from ortools.linear_solver import pywraplp
from collections import defaultdict


class LineupBuilder:
    def __init__(self, solver, player_vars, mode="dk"):
        self.solver = solver
        self.player_vars = player_vars
        self.mode = mode
        self.stack_vars = {}  # will be assigned externally
        solver.SetNumThreads(6)
        # solver.SetSolverSpecificParametersAsString("cut_level:0")
        params = """
        cut_level: 0
        symmetry_level: 0
        interleave_search: false
        """

        #  stop_after_first_solution: true
        solver.SetSolverSpecificParametersAsString(params)

        # solver.EnableOutput()
        # solver.SetSolverSpecificParametersAsString("gomoryCuts off")
        # solver.SetSolverSpecificParametersAsString("mipCuts off")
        # solver.SetSolverSpecificParametersAsString("mipStrategies cutPasses 0")
        # solver.SetSolverSpecificParametersAsString("limits/solutions = 1")

    def apply_stack_hints(self, team_to_hitters, projections, top_n=5):
        def get_stack_projection(hitters):
            top_hitters = sorted(
                hitters, key=lambda h: projections.get(h.PLAYERID, 0), reverse=True
            )[:top_n]
            return sum(projections.get(h.PLAYERID, 0) for h in top_hitters)

        team_proj = {
            team: get_stack_projection(hitters)
            for team, hitters in team_to_hitters.items()
        }

        sorted_teams = sorted(team_proj.items(), key=lambda x: x[1])
        num_teams = len(sorted_teams)

        if num_teams <= 10:
            cutoff = int(num_teams * 0.4)
        elif num_teams <= 16:
            cutoff = int(num_teams * 0.5)
        elif num_teams <= 24:
            cutoff = int(num_teams * 0.6)
        else:
            cutoff = int(num_teams * 0.7)

        for i, team in enumerate(sorted_teams):
            if i < cutoff:
                y_5_var = self.stack_vars.get("y_5", {}).get(team)
                if y_5_var:
                    self.solver.SetHint([y_5_var], [0])
                y_3_var = self.stack_vars.get("y_3", {}).get(team)
                if y_3_var:
                    self.solver.SetHint([y_3_var], [0])
            else:
                y_5_var = self.stack_vars.get("y_5", {}).get(team)
                if y_5_var:
                    self.solver.SetHint([y_5_var], [1])
                y_3_var = self.stack_vars.get("y_3", {}).get(team)
                if y_3_var:
                    self.solver.SetHint([y_3_var], [1])

    def set_objective(self, i, projections_matrix, decay_multipliers=None):
        decay_multipliers = decay_multipliers or {}
        objective = self.solver.Objective()
        for (pid, pos), var in self.player_vars.items():
            fpts = projections_matrix[pid][i]
            decay = decay_multipliers.get(pid, 1.0)
            objective.SetCoefficient(var, fpts * decay)
        objective.SetMaximization()

    def solve(self, time_limit=None):
        if time_limit:
            self.solver.set_time_limit(time_limit)
        status = self.solver.Solve()
        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return None

        return [
            pid
            for (pid, pos), var in self.player_vars.items()
            if var.solution_value() > 0.5
        ]


def preprocess_player_vars(players_by_id):
    player_vars = {}
    team_to_vars = defaultdict(list)
    pos_to_vars = defaultdict(list)
    player_position_vars = defaultdict(list)
    pitcher_vars = []
    team_to_pitcher = {}

    solver = pywraplp.Solver.CreateSolver("SAT")
    # solver.SetSolverSpecificParametersAsString("num_search_workers:8\nmax_time_in_seconds:0.2")
    if not solver:
        raise RuntimeError("Could not create solver")

    for pid, row_list in players_by_id.items():
        for row in row_list:
            team = row.TEAM
            pos = row.POS
            pid = row.PLAYERID
            key = (pid, pos)
            var = solver.IntVar(0, 1, f"x_{pid}_{pos}")
            player_vars[key] = var

            pos_to_vars[pos].append(var)
            player_position_vars[pid].append(var)
            if pos == "P":
                pitcher_vars.append(var)
                team_to_pitcher[team] = var
                var.SetBranchingPriority(50)
            else:
                team_to_vars[team].append(var)

    return (
        solver,
        player_vars,
        team_to_vars,
        pos_to_vars,
        player_position_vars,
        pitcher_vars,
        team_to_pitcher,
    )


def build_solver_5_3(players_by_id, team_to_opp, salary_cap=50000):
    (
        solver,
        player_vars,
        team_to_vars,
        pos_to_vars,
        player_position_vars,
        pitcher_vars,
        team_to_pitcher,
    ) = preprocess_player_vars(players_by_id)

    solver.Add(solver.Sum(pitcher_vars) == 2)
    solver.Add(
        solver.Sum(
            var for pos, vars_ in pos_to_vars.items() if pos != "P" for var in vars_
        )
        == 8
    )

    solver.Add(
        solver.Sum(
            [
                row.SALARY * player_vars[(row.PLAYERID, row.POS)]
                for row_list in players_by_id.values()
                for row in row_list
            ]
        )
        <= salary_cap
    )

    required_pos = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}
    for pos, count in required_pos.items():
        if pos in pos_to_vars:
            solver.Add(solver.Sum(pos_to_vars[pos]) == count)

    for pid, vars_ in player_position_vars.items():
        solver.Add(solver.Sum(vars_) <= 1)

    y_5 = {}
    y_3 = {}
    for team, hitter_vars in team_to_vars.items():
        y_5[team] = solver.IntVar(0, 1, f"stack5_{team}")
        y_5[team].SetBranchingPriority(100)
        y_3[team] = solver.IntVar(0, 1, f"stack3_{team}")
        y_3[team].SetBranchingPriority(90)

        if team_to_opp[team] in team_to_pitcher:
            p_var = team_to_pitcher[team_to_opp[team]]
        else:
            p_var = 0

        solver.Add(y_5[team] + y_3[team] + p_var <= 1)

        team_stack_total = solver.Sum(hitter_vars)
        solver.Add(team_stack_total == 5 * y_5[team] + 3 * y_3[team])

    solver.Add(solver.Sum(y_5.values()) == 1)
    solver.Add(solver.Sum(y_3.values()) == 1)

    builder = LineupBuilder(solver, player_vars, "dk")
    builder.stack_vars = {"y_5": y_5, "y_3": y_3}
    return builder


def build_solver_5_2(players_by_id, team_to_opp, salary_cap=50000):
    (
        solver,
        player_vars,
        team_to_vars,
        pos_to_vars,
        player_position_vars,
        pitcher_vars,
        team_to_pitcher,
    ) = preprocess_player_vars(players_by_id)

    solver.Add(solver.Sum(pitcher_vars) == 2)
    solver.Add(
        solver.Sum(
            var for pos, vars_ in pos_to_vars.items() if pos != "P" for var in vars_
        )
        == 8
    )
    solver.Add(
        solver.Sum(
            [
                row.SALARY * player_vars[(row.PLAYERID, row.POS)]
                for row_list in players_by_id.values()
                for row in row_list
            ]
        )
        <= salary_cap
    )

    required_pos = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}
    for pos, count in required_pos.items():
        if pos in pos_to_vars:
            solver.Add(solver.Sum(pos_to_vars[pos]) == count)
    for pid, vars_ in player_position_vars.items():
        solver.Add(solver.Sum(vars_) <= 1)

    y_5 = {}
    y_2 = {}
    for team, hitter_vars in team_to_vars.items():
        y_5[team] = solver.IntVar(0, 1, f"stack5_{team}")
        y_2[team] = solver.IntVar(0, 1, f"stack2_{team}")

        team_total = solver.Sum(hitter_vars)
        solver.Add(team_total >= 5 * y_5[team])
        solver.Add(team_total <= 5)
        solver.Add(team_total >= 2 * y_2[team])

        if team_to_opp[team] in team_to_pitcher:
            solver.Add(y_5[team] + y_2[team] + team_to_pitcher[team_to_opp[team]] <= 1)
            for var in hitter_vars:
                solver.Add(var + team_to_pitcher[team_to_opp[team]] <= 1)

    solver.Add(solver.Sum(y_5.values()) == 1)
    solver.Add(solver.Sum(y_2.values()) == 1)

    builder = LineupBuilder(solver, player_vars, "dk")
    builder.stack_vars = {"y_5": y_5, "y_2": y_2}
    return builder


def build_solver_5(players_by_id, team_to_opp, salary_cap=50000):
    (
        solver,
        player_vars,
        team_to_vars,
        pos_to_vars,
        player_position_vars,
        pitcher_vars,
        team_to_pitcher,
    ) = preprocess_player_vars(players_by_id)
    solver.Add(solver.Sum(pitcher_vars) == 2)
    solver.Add(
        solver.Sum(
            var for pos, vars_ in pos_to_vars.items() if pos != "P" for var in vars_
        )
        == 8
    )
    solver.Add(
        solver.Sum(
            [
                row.SALARY * player_vars[(row.PLAYERID, row.POS)]
                for row_list in players_by_id.values()
                for row in row_list
            ]
        )
        <= salary_cap
    )

    required_pos = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}
    for pos, count in required_pos.items():
        if pos in pos_to_vars:
            solver.Add(solver.Sum(pos_to_vars[pos]) == count)
    for pid, vars_ in player_position_vars.items():
        solver.Add(solver.Sum(vars_) <= 1)

    y_5 = {}
    for team, hitter_vars in team_to_vars.items():
        y_5[team] = solver.IntVar(0, 1, f"stack5_{team}")

        team_total = solver.Sum(hitter_vars)
        solver.Add(team_total >= 5 * y_5[team])
        solver.Add(team_total <= (1 - y_5[team]) + (5 * y_5[team]))

        if team_to_opp[team] in team_to_pitcher:
            solver.Add(y_5[team] + team_to_pitcher[team_to_opp[team]] <= 1)
            for var in hitter_vars:
                solver.Add(team_to_pitcher[team_to_opp[team]] + var <= 1)

    solver.Add(solver.Sum(y_5.values()) == 1)

    builder = LineupBuilder(solver, player_vars, "dk")
    builder.stack_vars = {"y_5": y_5}
    return builder


def build_solver_4_4(players_by_id, team_to_opp, salary_cap=50000):
    (
        solver,
        player_vars,
        team_to_vars,
        pos_to_vars,
        player_position_vars,
        pitcher_vars,
        team_to_pitcher,
    ) = preprocess_player_vars(players_by_id)
    solver.Add(solver.Sum(pitcher_vars) == 2)
    solver.Add(
        solver.Sum(
            var for pos, vars_ in pos_to_vars.items() if pos != "P" for var in vars_
        )
        == 8
    )
    solver.Add(
        solver.Sum(
            [
                row.SALARY * player_vars[(row.PLAYERID, row.POS)]
                for row_list in players_by_id.values()
                for row in row_list
            ]
        )
        <= salary_cap
    )

    required_pos = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}
    for pos, count in required_pos.items():
        if pos in pos_to_vars:
            solver.Add(solver.Sum(pos_to_vars[pos]) == count)
    for pid, vars_ in player_position_vars.items():
        solver.Add(solver.Sum(vars_) <= 1)

    y_4a = {}
    y_4b = {}
    for team, hitter_vars in team_to_vars.items():
        y_4a[team] = solver.IntVar(0, 1, f"stack4a_{team}")
        y_4b[team] = solver.IntVar(0, 1, f"stack4b_{team}")

        team_total = solver.Sum(hitter_vars)
        solver.Add(team_total <= 4)
        solver.Add(team_total >= 4 * (y_4a[team] + y_4b[team]))
        solver.Add(y_4a[team] + y_4b[team] <= 1)

        if team_to_opp[team] in team_to_pitcher:
            solver.Add(
                y_4a[team] + y_4b[team] + team_to_pitcher[team_to_opp[team]] <= 1
            )
        else:
            solver.Add(y_4a[team] + y_4b[team] <= 1)

    solver.Add(solver.Sum(y_4a.values()) == 1)
    solver.Add(solver.Sum(y_4b.values()) == 1)

    builder = LineupBuilder(solver, player_vars, "dk")
    builder.stack_vars = {"y_4a": y_4a, "y_4b": y_4b}
    return builder


def build_solver_4_3_1(players_by_id, team_to_opp, salary_cap=50000):
    (
        solver,
        player_vars,
        team_to_vars,
        pos_to_vars,
        player_position_vars,
        pitcher_vars,
        team_to_pitcher,
    ) = preprocess_player_vars(players_by_id)
    solver.Add(solver.Sum(pitcher_vars) == 2)
    solver.Add(
        solver.Sum(
            var for pos, vars_ in pos_to_vars.items() if pos != "P" for var in vars_
        )
        == 8
    )
    solver.Add(
        solver.Sum(
            [
                row.SALARY * player_vars[(row.PLAYERID, row.POS)]
                for row_list in players_by_id.values()
                for row in row_list
            ]
        )
        <= salary_cap
    )

    required_pos = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}
    for pos, count in required_pos.items():
        if pos in pos_to_vars:
            solver.Add(solver.Sum(pos_to_vars[pos]) == count)
    for pid, vars_ in player_position_vars.items():
        solver.Add(solver.Sum(vars_) <= 1)

    y_4 = {}
    y_3 = {}
    for team, hitter_vars in team_to_vars.items():
        y_4[team] = solver.IntVar(0, 1, f"stack4_{team}")
        y_3[team] = solver.IntVar(0, 1, f"stack3_{team}")

        team_total = solver.Sum(hitter_vars)
        solver.Add(team_total >= 4 * y_4[team] + 3 * y_3[team])
        solver.Add(
            team_total <= 4 * y_4[team] + 3 * y_3[team] + (1 - (y_4[team] + y_3[team]))
        )
        solver.Add(y_4[team] + y_3[team] <= 1)
        if team_to_opp[team] in team_to_pitcher:
            solver.Add(y_4[team] + y_3[team] + team_to_pitcher[team_to_opp[team]] <= 1)
            for var in hitter_vars:
                solver.Add(var + team_to_pitcher[team_to_opp[team]] <= 1)

    solver.Add(solver.Sum(y_4.values()) == 1)
    solver.Add(solver.Sum(y_3.values()) == 1)

    builder = LineupBuilder(solver, player_vars, "dk")
    builder.stack_vars = {"y_4": y_4, "y_3": y_3}
    return builder


def build_solver_4(players_by_id, team_to_opp, salary_cap=50000):
    (
        solver,
        player_vars,
        team_to_vars,
        pos_to_vars,
        player_position_vars,
        pitcher_vars,
        team_to_pitcher,
    ) = preprocess_player_vars(players_by_id)
    solver.Add(solver.Sum(pitcher_vars) == 2)
    solver.Add(
        solver.Sum(
            var for pos, vars_ in pos_to_vars.items() if pos != "P" for var in vars_
        )
        == 8
    )
    solver.Add(
        solver.Sum(
            [
                row.SALARY * player_vars[(row.PLAYERID, row.POS)]
                for row_list in players_by_id.values()
                for row in row_list
            ]
        )
        <= salary_cap
    )

    required_pos = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}
    for pos, count in required_pos.items():
        if pos in pos_to_vars:
            solver.Add(solver.Sum(pos_to_vars[pos]) == count)
    for pid, vars_ in player_position_vars.items():
        solver.Add(solver.Sum(vars_) <= 1)

    y_4 = {}
    for team, hitter_vars in team_to_vars.items():
        y_4[team] = solver.IntVar(0, 1, f"stack5_{team}")

        team_total = solver.Sum(hitter_vars)
        solver.Add(team_total >= 4 * y_4[team])
        solver.Add(team_total <= (4 * y_4[team] + 2 * (1 - y_4[team])))

        if team_to_opp[team] in team_to_pitcher:
            solver.Add(y_4[team] + team_to_pitcher[team_to_opp[team]] <= 1)
            for var in hitter_vars:
                solver.Add(team_to_pitcher[team_to_opp[team]] + var <= 1)

    solver.Add(solver.Sum(y_4.values()) == 1)

    builder = LineupBuilder(solver, player_vars, "dk")
    builder.stack_vars = {"y_4": y_4}
    return builder


def build_solver_3_3_1_1(players_by_id, team_to_opp, salary_cap=50000):
    (
        solver,
        player_vars,
        team_to_vars,
        pos_to_vars,
        player_position_vars,
        pitcher_vars,
        team_to_pitcher,
    ) = preprocess_player_vars(players_by_id)
    solver.Add(solver.Sum(pitcher_vars) == 2)
    solver.Add(
        solver.Sum(
            var for pos, vars_ in pos_to_vars.items() if pos != "P" for var in vars_
        )
        == 8
    )
    solver.Add(
        solver.Sum(
            [
                row.SALARY * player_vars[(row.PLAYERID, row.POS)]
                for row_list in players_by_id.values()
                for row in row_list
            ]
        )
        <= salary_cap
    )

    required_pos = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}
    for pos, count in required_pos.items():
        if pos in pos_to_vars:
            solver.Add(solver.Sum(pos_to_vars[pos]) == count)
    for pid, vars_ in player_position_vars.items():
        solver.Add(solver.Sum(vars_) <= 1)

    y_3a = {}
    y_3b = {}
    for team, hitter_vars in team_to_vars.items():
        y_3a[team] = solver.IntVar(0, 1, f"stack3a_{team}")
        y_3b[team] = solver.IntVar(0, 1, f"stack3b_{team}")

        team_total = solver.Sum(hitter_vars)
        solver.Add(team_total >= 3 * (y_3a[team] + y_3b[team]))
        solver.Add(
            team_total <= 3 * (y_3a[team] + y_3b[team]) + (1 - y_3a[team] - y_3b[team])
        )

        if team_to_opp[team] in team_to_pitcher:
            solver.Add(
                y_3a[team] + y_3b[team] + team_to_pitcher[team_to_opp[team]] <= 1
            )
            for var in hitter_vars:
                solver.Add(var + team_to_pitcher[team_to_opp[team]] <= 1)

    solver.Add(solver.Sum(y_3a.values()) == 1)
    solver.Add(solver.Sum(y_3b.values()) == 1)

    builder = LineupBuilder(solver, player_vars, "dk")
    builder.stack_vars = {"y_3a": y_3a, "y_3b": y_3b}
    return builder


def build_solver_chaos(players_by_id, team_to_opp, salary_cap=50000):
    (
        solver,
        player_vars,
        team_to_vars,
        pos_to_vars,
        player_position_vars,
        pitcher_vars,
        team_to_pitcher,
    ) = preprocess_player_vars(players_by_id)
    solver.Add(solver.Sum(pitcher_vars) == 2)
    solver.Add(
        solver.Sum(
            var for pos, vars_ in pos_to_vars.items() if pos != "P" for var in vars_
        )
        == 8
    )
    solver.Add(
        solver.Sum(
            [
                row.SALARY * player_vars[(row.PLAYERID, row.POS)]
                for row_list in players_by_id.values()
                for row in row_list
            ]
        )
        <= salary_cap
    )

    required_pos = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3}
    for pos, count in required_pos.items():
        if pos in pos_to_vars:
            solver.Add(solver.Sum(pos_to_vars[pos]) == count)
    for pid, vars_ in player_position_vars.items():
        solver.Add(solver.Sum(vars_) <= 1)

    for team, hitter_vars in team_to_vars.items():
        team_total = solver.Sum(hitter_vars)
        solver.Add(team_total <= 5)
        for var in hitter_vars:
            if team_to_opp[team] in team_to_pitcher:
                solver.Add(var + team_to_pitcher[team_to_opp[team]] <= 1)

    builder = LineupBuilder(solver, player_vars, "dk")
    builder.stack_vars = {}
    return builder
