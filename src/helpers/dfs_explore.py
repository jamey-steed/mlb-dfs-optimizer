# dfs_explore.py
# One-shot player/team exploratory analysis for MLB DFS sims, using tuple-style player_meta:
# player_meta[player_id] = (position, team, projection, salary, name, opp)

import os
import math
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional

# ----- Tuple indices -----
POS_I, TEAM_I, PROJ_I, SAL_I, NAME_I, OPP_I = 0, 1, 2, 3, 4, 5


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _meta(player_meta: Dict, pid) -> Tuple[str, str, float, float, str, Optional[str]]:
    pos, team, proj, sal, name, opp = player_meta[pid]
    return pos, team, proj, sal, name, opp


def _is_pitcher(pos: str) -> bool:
    # Robust: treat any position string containing 'P' as pitcher (e.g., 'P', 'SP', 'RP', 'P/UTIL')
    return "P" in str(pos).upper()


def _q(a: np.ndarray, q: float) -> float:
    return float(np.quantile(a, q))


def _cvar(a: np.ndarray, alpha: float = 0.9) -> float:
    t = np.quantile(a, alpha)
    m = a >= t
    return float(a[m].mean() if m.any() else t)


# ---------- Player-level stats ----------


def summarize_players(
    sims: np.ndarray, id_to_index: Dict, player_meta: Dict
) -> pd.DataFrame:
    S, P = sims.shape
    idx2id = {idx: pid for pid, idx in id_to_index.items()}
    rows = []
    for idx in range(P):
        pid = idx2id[idx]
        pos, team, proj, sal, name, opp = _meta(player_meta, pid)
        x = sims[:, idx]
        mean = float(x.mean())
        var = float(x.var(ddof=1))
        std = math.sqrt(var) if var > 0 else 0.0
        c = x - mean
        m2 = float((c**2).mean())
        m3 = float((c**3).mean())
        m4 = float((c**4).mean())
        skew = m3 / (m2**1.5 + 1e-12) if m2 > 0 else 0.0
        kurt = m4 / (m2**2 + 1e-12) if m2 > 0 else 0.0
        rec = {
            "id": pid,
            "name": name,
            "team": team,
            "opp": opp,
            "pos": pos,
            "is_pitcher": _is_pitcher(pos),
            "salary": sal,
            "projection": proj,
            "mean": mean,
            "std": std,
            "var": var,
            "skew": skew,
            "kurtosis": kurt,
            "p90": _q(x, 0.90),
            "p95": _q(x, 0.95),
            "p99": _q(x, 0.99),
            "cvar90": _cvar(x, 0.90),
            "cvar95": _cvar(x, 0.95),
        }
        if proj is not None:
            rec["mean_vs_proj"] = mean - float(proj)
            rec["mean_over_proj"] = (mean / float(proj)) if float(proj) > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------- Covariance / Correlation ----------


def cov_corr_matrices(sims: np.ndarray):
    cov = np.cov(sims, rowvar=False)
    std = np.sqrt(np.maximum(np.diag(cov), 0))
    denom = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    # Spearman via ranks (no SciPy)
    ranks = np.argsort(np.argsort(sims, axis=0), axis=0).astype(float) + 1.0
    ranks /= sims.shape[0]
    corr_spearman = np.corrcoef(ranks, rowvar=False)
    # Ledoit–Wolf shrinkage
    lw = LedoitWolf().fit(sims)
    cov_shrunk = lw.covariance_
    std_s = np.sqrt(np.maximum(np.diag(cov_shrunk), 0))
    denom_s = np.outer(std_s, std_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_shrunk = np.where(denom_s > 0, cov_shrunk / denom_s, 0.0)
    return cov, corr, cov_shrunk, corr_shrunk, corr_spearman


# ---------- Team totals & correlation ----------


def team_totals(sims: np.ndarray, id_to_index: Dict, player_meta: Dict):
    team2cols: Dict[str, List[int]] = {}
    for pid, idx in id_to_index.items():
        pos, team, proj, sal, name, opp = _meta(player_meta, pid)
        if not team or _is_pitcher(pos):
            continue
        team2cols.setdefault(team, []).append(idx)
    teams = sorted(team2cols.keys())
    cols_list = [team2cols[t] for t in teams]
    team_arr = (
        np.column_stack([sims[:, cols].sum(axis=1) for cols in cols_list])
        if teams
        else np.zeros((sims.shape[0], 0))
    )
    return teams, team_arr


def summarize_team_totals(team_arr: np.ndarray, teams: List[str]) -> pd.DataFrame:
    rows = []
    for j, t in enumerate(teams):
        x = team_arr[:, j]
        mean = float(x.mean())
        var = float(x.var(ddof=1))
        std = math.sqrt(var) if var > 0 else 0.0
        c = x - mean
        m2 = float((c**2).mean())
        m3 = float((c**3).mean())
        m4 = float((c**4).mean())
        skew = m3 / (m2**1.5 + 1e-12) if m2 > 0 else 0.0
        kurt = m4 / (m2**2 + 1e-12) if m2 > 0 else 0.0
        rows.append(
            {
                "team": t,
                "mean": mean,
                "std": std,
                "var": var,
                "p90": _q(x, 0.90),
                "p95": _q(x, 0.95),
                "p99": _q(x, 0.99),
                "cvar90": _cvar(x, 0.90),
                "cvar95": _cvar(x, 0.95),
                "skew": skew,
                "kurtosis": kurt,
            }
        )
    return pd.DataFrame(rows)


def team_corr_matrix(team_arr: np.ndarray) -> np.ndarray:
    return (
        np.corrcoef(team_arr, rowvar=False) if team_arr.shape[1] else np.zeros((0, 0))
    )


# ---------- Same-game team correlation ----------


def same_game_team_corr_table(
    teams: List[str], team_arr: np.ndarray, id_to_index: Dict, player_meta: Dict
) -> pd.DataFrame:
    """One row per game (unordered pair {team, opp}) with correlation of team totals."""
    if team_arr.shape[1] == 0:
        return pd.DataFrame(columns=["team", "opp", "corr"])

    team_to_col = {t: i for i, t in enumerate(teams)}

    games = set()
    for pid, _ in id_to_index.items():
        pos, team, proj, sal, name, opp = _meta(player_meta, pid)
        if team and opp and not _is_pitcher(pos):
            a, b = sorted([team, opp])
            games.add((a, b))

    rows = []
    for a, b in sorted(games):
        ia, ib = team_to_col.get(a), team_to_col.get(b)
        if ia is None or ib is None:
            continue
        c = float(np.corrcoef(team_arr[:, ia], team_arr[:, ib])[0, 1])
        rows.append({"team": a, "opp": b, "corr": c})
    return pd.DataFrame(rows).sort_values("corr", ascending=False)


# ---------- Pitcher / Hitter opposing correlation ----------


def pitcher_idx_by_team(
    sims: np.ndarray, id_to_index: Dict, player_meta: Dict
) -> Dict[str, int]:
    """Pick a starting pitcher index per team (prefer SP; tiebreak by mean)."""
    cand: Dict[str, List[Tuple[int, str]]] = {}
    for pid, idx in id_to_index.items():
        pos, team, proj, sal, name, opp = _meta(player_meta, pid)
        if team and _is_pitcher(pos):
            cand.setdefault(team, []).append((idx, pos))
    team_to_pidx: Dict[str, int] = {}
    for team, lst in cand.items():
        if not lst:
            continue
        sps = [idx for idx, pos in lst if "SP" in str(pos).upper()]
        pool = sps if sps else [idx for idx, _ in lst]
        if len(pool) > 1:
            means = sims[:, pool].mean(axis=0)
            team_to_pidx[team] = pool[int(np.argmax(means))]
        else:
            team_to_pidx[team] = pool[0]
    return team_to_pidx


def pitcher_vs_opp_hitters_corr(
    sims: np.ndarray, id_to_index: Dict, player_meta: Dict
) -> pd.DataFrame:
    """For each pitcher, corr vs every opposing hitter (hitter.opp == pitcher.team)."""
    hitters_by_opp: Dict[str, List[int]] = {}
    hitters_pid_by_opp: Dict[str, List] = {}
    for pid, idx in id_to_index.items():
        pos, team, proj, sal, name, opp = _meta(player_meta, pid)
        if opp and not _is_pitcher(pos):
            hitters_by_opp.setdefault(opp, []).append(idx)
            hitters_pid_by_opp.setdefault(opp, []).append(pid)

    rows = []
    for pid_p, idx_p in id_to_index.items():
        pos, p_team, proj, sal, p_name, p_opp = _meta(player_meta, pid_p)
        if not _is_pitcher(pos) or not p_team:
            continue
        cols = hitters_by_opp.get(p_team, [])
        if not cols:
            continue
        sp = sims[:, idx_p][:, None]
        bats = sims[:, cols]
        mat = np.corrcoef(np.hstack([sp, bats]), rowvar=False)
        vec = mat[0, 1:]
        rows.append(
            {
                "pitcher_id": pid_p,
                "pitcher": p_name,
                "team": p_team,
                "n_opp_hitters": int(len(vec)),
                "avg_corr_vs_opp_hitters": float(np.nanmean(vec)),
                "min_corr": float(np.nanmin(vec)),
                "max_corr": float(np.nanmax(vec)),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_corr_vs_opp_hitters")


def hitter_vs_opp_sp_corr(
    sims: np.ndarray, id_to_index: Dict, player_meta: Dict
) -> pd.DataFrame:
    """For each hitter, corr with the opposing team's starting pitcher."""
    pidx_by_team = pitcher_idx_by_team(sims, id_to_index, player_meta)
    rows = []
    for pid_h, idx_h in id_to_index.items():
        pos, team, proj, sal, name, opp = _meta(player_meta, pid_h)
        if _is_pitcher(pos) or not opp:
            continue
        sp_idx = pidx_by_team.get(opp)
        if sp_idx is None:
            continue
        c = float(np.corrcoef(sims[:, idx_h], sims[:, sp_idx])[0, 1])
        rows.append(
            {"id": pid_h, "name": name, "team": team, "opp": opp, "corr_vs_opp_sp": c}
        )
    return pd.DataFrame(rows).sort_values("corr_vs_opp_sp")


# ---------- Player vs own-team total corr ----------


def player_corr_with_team_total(
    sims: np.ndarray,
    id_to_index: Dict,
    player_meta: Dict,
    teams: List[str],
    team_arr: np.ndarray,
) -> pd.DataFrame:
    team_to_col = {t: i for i, t in enumerate(teams)}
    rows = []
    for pid, idx in id_to_index.items():
        pos, tm, proj, sal, name, opp = _meta(player_meta, pid)
        if not tm:
            continue
        j = team_to_col.get(tm)
        if j is None:
            continue
        c = float(np.corrcoef(sims[:, idx], team_arr[:, j])[0, 1])
        rows.append(
            {"id": pid, "name": name, "team": tm, "pos": pos, "corr_with_team_total": c}
        )
    return pd.DataFrame(rows).sort_values("corr_with_team_total", ascending=False)


# ---------- Stack synergy (variance amplification) ----------


def synergy_for_team(
    sims: np.ndarray,
    id_to_index: Dict,
    player_meta: Dict,
    team: str,
    k: int,
    top_n: int = 9,
) -> pd.DataFrame:
    # pick top_n team hitters by mean; evaluate all k-combos for synergy ratio
    idxs, ids = [], []
    for pid, idx in id_to_index.items():
        pos, tm, proj, sal, name, opp = _meta(player_meta, pid)
        if tm == team and not _is_pitcher(pos):
            idxs.append(idx)
            ids.append(pid)
    if not idxs:
        return pd.DataFrame(
            columns=["team", "k", "players", "var_stack", "var_sum", "synergy_ratio"]
        )
    means = sims[:, idxs].mean(axis=0)
    order = np.argsort(-means)[: min(top_n, len(idxs))]
    sel_cols = [idxs[i] for i in order]
    sel_ids = [ids[i] for i in order]
    sub = sims[:, sel_cols]
    var_all = sub.var(axis=0, ddof=1)
    rows = []
    for comb in combinations(range(sub.shape[1]), k):
        block = sub[:, comb].sum(axis=1)
        var_stack = float(block.var(ddof=1))
        var_sum = float(var_all[list(comb)].sum())
        rows.append(
            {
                "team": team,
                "k": k,
                "players": tuple(sel_ids[i] for i in comb),
                "var_stack": var_stack,
                "var_sum": var_sum,
                "synergy_ratio": var_stack / (var_sum + 1e-12),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("synergy_ratio", ascending=False)
        .reset_index(drop=True)
    )


def synergy_all_teams(
    sims: np.ndarray, id_to_index: Dict, player_meta: Dict, k: int, top_n: int = 9
):
    teams = sorted(
        {
            _meta(player_meta, pid)[TEAM_I]
            for pid in id_to_index
            if not _is_pitcher(_meta(player_meta, pid)[POS_I])
        }
    )
    frames = []
    for t in teams:
        if not t:
            continue
        df = synergy_for_team(sims, id_to_index, player_meta, t, k=k, top_n=top_n)
        frames.append(df.head(20))
    if not frames:
        return pd.DataFrame(
            columns=["team", "k", "players", "var_stack", "var_sum", "synergy_ratio"]
        )
    out = pd.concat(frames, ignore_index=True)
    best = (
        out.groupby("team")["synergy_ratio"]
        .max()
        .reset_index()
        .rename(columns={"synergy_ratio": "best_synergy_ratio"})
    )
    return out, best


# ---------- PCA ----------


def pca_factors(sims: np.ndarray, n_components: int = 8):
    X = sims - sims.mean(axis=0, keepdims=True)
    pca = PCA(n_components=min(n_components, X.shape[1]))
    scores = pca.fit_transform(X)
    return pca, scores, pca.explained_variance_ratio_, pca.components_


def top_pca_loadings(
    loadings: np.ndarray, id_to_index: Dict, player_meta: Dict, top_k: int = 12
) -> pd.DataFrame:
    idx2id = {idx: pid for pid, idx in id_to_index.items()}
    K, P = loadings.shape
    rows = []
    for k in range(K):
        w = loadings[k, :]
        pos_order = np.argsort(-w)[:top_k]
        neg_order = np.argsort(w)[:top_k]
        for sign, order in [("+", pos_order), ("-", neg_order)]:
            for idx in order:
                pid = idx2id[idx]
                pos, team, proj, sal, name, opp = _meta(player_meta, pid)
                rows.append(
                    {
                        "factor": k,
                        "sign": sign,
                        "id": pid,
                        "name": name,
                        "team": team,
                        "pos": pos,
                        "loading": float(w[idx]),
                    }
                )
    return pd.DataFrame(rows)


# ---------- Main entry ----------


def run_exploration(
    sims: np.ndarray, id_to_index: Dict, player_meta: Dict, out_dir: str = "explore_out"
):
    _ensure_dir(out_dir)

    # 1) Player stats
    players = summarize_players(sims, id_to_index, player_meta)
    players.to_csv(os.path.join(out_dir, "player_stats.csv"), index=False)

    # 2) Cov/Cor
    cov, corr, cov_s, corr_s, corr_spear = cov_corr_matrices(sims)
    np.save(os.path.join(out_dir, "cov_raw.npy"), cov)
    np.save(os.path.join(out_dir, "cov_shrunk.npy"), cov_s)
    np.save(os.path.join(out_dir, "corr_pearson.npy"), corr)
    np.save(os.path.join(out_dir, "corr_shrunk.npy"), corr_s)
    np.save(os.path.join(out_dir, "corr_spearman.npy"), corr_spear)

    # 3) Team totals & correlations
    teams, team_arr = team_totals(sims, id_to_index, player_meta)
    team_stats = summarize_team_totals(team_arr, teams)
    team_stats.to_csv(os.path.join(out_dir, "team_totals_stats.csv"), index=False)
    if teams:
        pd.DataFrame(
            np.corrcoef(team_arr, rowvar=False), index=teams, columns=teams
        ).to_csv(os.path.join(out_dir, "team_corr.csv"))

        # Same-game team correlation table
        same_game = same_game_team_corr_table(teams, team_arr, id_to_index, player_meta)
        if not same_game.empty:
            same_game.to_csv(
                os.path.join(out_dir, "same_game_team_corr.csv"), index=False
            )

        # Player vs own team-total corr
        pct = player_corr_with_team_total(
            sims, id_to_index, player_meta, teams, team_arr
        )
        if not pct.empty:
            pct.to_csv(os.path.join(out_dir, "player_corr_with_team.csv"), index=False)

    # 4) Pitcher vs opp hitters & Hitter vs opp SP
    sp_opp = pitcher_vs_opp_hitters_corr(sims, id_to_index, player_meta)
    if not sp_opp.empty:
        sp_opp.to_csv(os.path.join(out_dir, "sp_vs_opp_hitters_corr.csv"), index=False)

    hvsp = hitter_vs_opp_sp_corr(sims, id_to_index, player_meta)
    if not hvsp.empty:
        hvsp.to_csv(os.path.join(out_dir, "hitter_vs_opp_sp_corr.csv"), index=False)

    # 5) Synergy (k=4,5)
    for k in (4, 5):
        res = synergy_all_teams(sims, id_to_index, player_meta, k=k, top_n=9)
        if isinstance(res, tuple):
            combos, best = res
            combos.to_csv(os.path.join(out_dir, f"synergy_k{k}.csv"), index=False)
            best.sort_values("best_synergy_ratio", ascending=False).to_csv(
                os.path.join(out_dir, f"synergy_k{k}_best_by_team.csv"), index=False
            )
        else:
            res.to_csv(os.path.join(out_dir, f"synergy_k{k}.csv"), index=False)

    # 6) PCA overview
    pca, scores, explained, loadings = pca_factors(sims, n_components=8)
    pd.DataFrame(
        {"factor": range(len(explained)), "explained_variance_ratio": explained}
    ).to_csv(os.path.join(out_dir, "pca_explained.csv"), index=False)
    top_loads = top_pca_loadings(loadings, id_to_index, player_meta, top_k=12)
    top_loads.to_csv(os.path.join(out_dir, "pca_top_loadings.csv"), index=False)

    return {
        "player_stats": os.path.join(out_dir, "player_stats.csv"),
        "team_stats": os.path.join(out_dir, "team_totals_stats.csv"),
        "team_corr": os.path.join(out_dir, "team_corr.csv") if teams else None,
        "same_game_team_corr": (
            os.path.join(out_dir, "same_game_team_corr.csv") if teams else None
        ),
        "player_corr_with_team": (
            os.path.join(out_dir, "player_corr_with_team.csv") if teams else None
        ),
        "sp_vs_opp_hitters_corr": (
            os.path.join(out_dir, "sp_vs_opp_hitters_corr.csv")
            if not sp_opp.empty
            else None
        ),
        "hitter_vs_opp_sp_corr": (
            os.path.join(out_dir, "hitter_vs_opp_sp_corr.csv")
            if not hvsp.empty
            else None
        ),
        "synergy_k4": os.path.join(out_dir, "synergy_k4.csv"),
        "synergy_k5": os.path.join(out_dir, "synergy_k5.csv"),
        "pca_explained": os.path.join(out_dir, "pca_explained.csv"),
        "pca_top_loadings": os.path.join(out_dir, "pca_top_loadings.csv"),
    }


# Example usage:
# res = run_exploration(sims, id_to_index, player_meta, out_dir="explore_out")
