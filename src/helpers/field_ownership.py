from collections import Counter
import numpy as np


class FieldOwnershipTracker:
    def __init__(
        self,
        total_lineups,
        stack_shape_targets=None,
        team_stack_targets=None,
        pitcher_targets=None,
    ):
        self.total_lineups = total_lineups

        self.shape_counter = Counter()
        self.team_stack_counter = Counter()
        self.mini_stack_counter = Counter()
        self.pitcher_counter = Counter()

        self.stack_shape_targets = stack_shape_targets or {}
        self.team_stack_targets = team_stack_targets or {}
        self.pitcher_targets = pitcher_targets or {}
        print(self.pitcher_targets)

    def update_stack_shape(self, shape):
        self.shape_counter[shape] += 1

    def update_primary_team(self, team):
        self.team_stack_counter[team] += 1

    def update_secondary_team(self, team):
        self.mini_stack_counter[team] += 1

    def update_pitchers(self, pitcher_ids):
        for pid in pitcher_ids:
            self.pitcher_counter[pid] += 1

    def get_stack_shape_penalty(self, shape):
        current = self.shape_counter.get(shape, 0)
        target = self.stack_shape_targets.get(
            shape, 1 / max(1, len(self.stack_shape_targets))
        )
        expected = target * self.total_lineups
        return self._penalty(current, expected)

    def get_team_stack_penalty(self, team):
        current = self.team_stack_counter.get(team, 0)
        target = self.team_stack_targets.get(
            team, 1 / max(1, len(self.team_stack_targets))
        )
        expected = target * self.total_lineups
        return self._penalty(current, expected)

    def get_pitcher_penalty(self, pitcher_id):
        current = self.pitcher_counter.get(pitcher_id, 0)
        target = self.pitcher_targets.get(
            pitcher_id, 1 / max(1, len(self.pitcher_targets))
        )
        expected = target * self.total_lineups
        return self._penalty(current, expected)

    def _penalty(self, current, expected, alpha=1.0):
        """
        Log-based symmetric penalty function.
        Encourages usage to move toward expected share smoothly.

        Returns >1.0 if underused, <1.0 if overused.
        """
        if expected == 0:
            return 1.0  # no target specified — no penalty

        deviation = (current - expected) / expected
        return np.exp(-alpha * deviation)
