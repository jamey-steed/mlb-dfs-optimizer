from collections import Counter, defaultdict


class Lineup:
    def __init__(self, player_ids, positions, metadata, features=None):
        self.player_ids = player_ids  # List[str] — just the player IDs
        self.positions = positions  # Dict[str, str] — PLAYERID -> position used
        self.metadata = metadata  # Dict[str, Tuple] — player metadata dict
        self._features = features or {}
        self.ev = None
        self.frozen_ids = frozenset(sorted(self.player_ids))

    @classmethod
    def from_raw_list(cls, player_ids, metadata):
        """
        Create a Lineup object from a list of player IDs and metadata.
        Uses greedy position assignment.
        """
        # print("------------------------------------------------------------------")
        required_positions = {
            "P": 2,
            "C": 1,
            "1B": 1,
            "2B": 1,
            "3B": 1,
            "SS": 1,
            "OF": 3,
        }
        position_slots = []
        for pos, count in required_positions.items():
            position_slots.extend([f"{pos}-{i}" for i in range(count)])

        # Build reverse: position slot → base position
        slot_to_pos = {slot: slot.split("-")[0] for slot in position_slots}

        # Build pid → eligible slots
        pid_to_slots = {}
        for pid in player_ids:
            # print(metadata[pid][0])
            # print(metadata[pid][-1])
            eligible = metadata[pid][0].split("/")  # e.g., "1B/OF"
            pid_to_slots[pid] = [
                slot for slot in position_slots if slot_to_pos[slot] in eligible
            ]

        # Backtracking assignment
        assignment = {}
        used_slots = set()

        def backtrack(index):
            if index == len(player_ids):
                return True
            pid = player_ids[index]
            for slot in pid_to_slots[pid]:
                if slot not in used_slots:
                    assignment[pid] = slot_to_pos[slot]
                    used_slots.add(slot)
                    if backtrack(index + 1):
                        return True
                    del assignment[pid]
                    used_slots.remove(slot)
            return False

        if not backtrack(0):
            for pid in player_ids:
                print(metadata[pid])
            raise ValueError("No valid position assignment found")

        return cls(player_ids=player_ids, positions=assignment, metadata=metadata)

    @property
    def features(self):
        if not self._features:
            self._features = self.compute_features()
        return self._features

    def compute_features(self):
        pitchers, hitters = [], []
        team_counts = Counter()
        opp_pitcher_teams = set()

        for pid in self.player_ids:
            meta = self.metadata[pid]
            if self.positions.get(pid) == "P":
                pitchers.append(pid)
                opp_pitcher_teams.add(meta[5])
            else:
                hitters.append(pid)
                team_counts[meta[1]] += 1

        stacks = team_counts.most_common()
        stack_teams = []
        stack_shape = []
        max_value = 0
        primary_stacks = []
        for team, value in stacks:
            if value > max_value:
                primary_stacks = [team]
            elif value == max_value:
                primary_stacks.append(team)
            stack_teams.append(team)
            stack_shape.append(value)

        return {
            "pitchers": pitchers,
            "pitcher_teams": [self.metadata[p][6] for p in pitchers],
            "hitters": hitters,
            "stack_teams": stack_teams,
            "stack_shape": tuple(stack_shape),
            "num_unique_teams": len(team_counts),
            "total_salary": sum(
                self.metadata[pid][5]
                for pid in self.player_ids
                if self.metadata[pid][5] is not None
            ),
            "primary_stack_teams": primary_stacks,
        }

    def contains_team(self, team):
        return team in self.features["stack_teams"]

    def is_hitter_opposing_pitcher(self, team):
        return team in self.features["pitcher_teams"]

    def get_stack_size(self):
        shape = self.features["stack_shape"]
        return max(shape) if shape else 0

    def get_stack_shape(self):
        return self.features["stack_shape"]

    def get_ev(self):
        if self.ev:
            return self.ev
        else:
            return None

    def copy(self):
        return Lineup(
            player_ids=self.player_ids[:],
            positions=self.positions.copy(),
            metadata=self.metadata,
            features=self._features.copy(),
        )

    def __repr__(self):
        shape = self.features["stack_shape"]
        position_order = {"C": 1, "1B": 2, "2B": 3, "3B": 4, "SS": 5, "OF": 6}
        self.features["hitters"].sort(key=lambda x: position_order[self.positions[x]])
        hitter_str = "".join(
            [f"{self.metadata[pid][-1]} ({pid})," for pid in self.features["hitters"]]
        )
        return f"{int(self.features['total_salary'])};{self.get_ev()};{shape};{self.features['stack_teams']};{self.metadata[self.features['pitchers'][0]][-1]} ({self.features['pitchers'][0]}),{self.metadata[self.features['pitchers'][1]][-1]} ({self.features['pitchers'][1]}),{hitter_str}"
