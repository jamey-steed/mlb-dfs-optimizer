# ⚾ DraftKings MLB DFS Simulation & Optimization Engine

> **License Notice:**  
> This repository is licensed under the [Apache 2.0 License](./LICENSE).  
> It is provided strictly for educational and demonstrative purposes.  
> **Commercial use, redistribution, or monetization of this code or derivatives is discouraged without written permission.**

## Overview

This repository contains a **simulation and optimization framework** for DraftKings MLB daily fantasy sports (DFS).
It demonstrates large-scale lineup generation, ownership-aware field modeling, and basic Monte Carlo–based game simulations.

Originally part of a larger proprietary system, this version has been **abstracted and sanitized** to not reveal any private data sources or strategy logic, so I would not recommend using it in production. Additionally, the actual lineup/field optimization step is done through AWS lambda fanout, described in detail below.

---

## 🧠 Core Components

| Module                           | Description                                                                                                                                                                                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `game_sims.py`                   | Simulates MLB games at the plate-appearance level using probabilistic outcomes (walks, singles, doubles, home runs, etc.) under DraftKings scoring. Produces fantasy point distributions and pitcher–hitter correlations.                         |
| `int_builder.py`                 | Builds valid DraftKings lineups using **OR-Tools** mixed integer programming. Supports multiple stack structures (`5-3`, `5-2`, `4-4`, etc.) and enforces roster constraints such as salary cap, positions, and no hitters vs. opposing pitchers. |
| `lineup.py`                      | Encapsulates a lineup as a reusable object with computed features (stack shape, salary, team counts, etc.) and helper utilities for lineup comparison and display.                                                                                |
| `helpers/helper.py`              | Utility functions for projection sampling, standard deviation estimation, stack analysis, and ownership decay logic. Provides both scalar and vectorized versions for performance.                                                                |
| `build_test.py`                  | Entry point that orchestrates data loading, solver creation, lineup generation, and ownership-tracking logic. Supports hybrid builds, Lambda-based fanouts, and field modeling.                                                                   |
| `helpers/field_ownership.py`     | Tracks global field ownership across players, pitchers, and stack types. Applies penalties via decay to encourage diversity and realistic field distributions.                                                                                    |
| `helpers/stack_target_loader.py` | Loads precomputed team and pitcher ownership targets used to steer lineup diversity.                                                                                                                                                              |

---

## 🌇 Project Structure

```
.
├── requirements.txt
├── .gitignore
└── src/
    ├── build_test.py
    ├── build_test_lambda.py - not used in this example but included to showcase what the lambda call(s) might look like
    ├── game_sims.py
    ├── int_builder.py
    ├── lineup.py
    ├── helpers/
    │   ├── helper.py
    │   ├── dfs_explore.py
    │   ├── field_ownership.py
    │   ├── stack_target_loader.py
    │   ├── TEAM_NAME_TRANSLATION.py
    ├── explore_out/
    └── output_data/
```

---

## ⚙️ How It Works

1. **Load Data**
   Reads sample hitters and pitchers CSVs (under `test_slates/`) and cleans them into a consistent schema.

2. **Build Metadata**
   Creates player metadata including position eligibility, salary, team, and projections, along with derived ceiling/floor statistics.

3. **Construct Solvers**
   Uses OR-Tools to build constraint-based lineup solvers for each stack shape. This could be replaced with the AWS lambda architecture below.

4. **Generate Lineups**
   Iteratively samples projections, applies ownership decay, and solves each lineup with the appropriate solver.
   Field ownership is tracked throughout to match desired stack and player distributions.

5. **Simulate Games**
   Run `game_sims.py` to simulate games, producing distributions of fantasy points for hitters and pitchers. This can also be done independently of the full pipeline.

6. **Analyze Results**
   Output lineups and summary data are written to `explore_out/` or `output_data/` for downstream use.

---

## ☁️ AWS Lambda Fanout Architecture

This project supports **fully parallelized lineup generation** via an AWS Lambda–based architecture designed for scalability and cost efficiency. These lambdas are not currently included in this repo.

### Architecture Overview

The distributed build process consists of two coordinated Lambda functions:

1. **Fanout Lambda**

   * Orchestrates lineup generation by splitting the total job into batches.
   * Invokes hundreds/thousands of `solve_lineup_lambda` functions concurrently using asynchronous `boto3.client('lambda').invoke()` calls.
   * Uses thread pooling for efficient concurrent invocations and error handling.
   * Collects and aggregates the lineups from Solver Lambda and either writes them to S3 or returns them inline, depending on the size.

2. **Solver Lambda (`solve_lineup_lambda`)**

   * Each instance initializes an OR-Tools solver and builds one or more lineups.
   * Reads and writes to a shared **Redis (ElastiCache)** instance to retrieve and update ownership/decay statistics in near real time.
   * Returns results directly to the fanout function for aggregation.

### Centralized Coordination

A **Redis/ElastiCache layer** serves as the coordination backbone:

* Tracks global ownership decay metrics and stack counts across all Lambda workers.
* Prevents over-selection (or under-selection) of players or teams across parallel invocations.
* Enables iterative, ownership-aware lineup generation without inter-Lambda blocking.

### Advantages

* 🚀 **Massive concurrency** — thousands of lineups can be solved in parallel within seconds.
* 💰 **Low cost** — each full build completes for under $1 at typical AWS Lambda pricing.
* 🔁 **Scalable** — automatically adapts to different slate sizes or stack configurations.
* 🧩 **Composable** — can integrate with AWS Step Functions or S3-based pipelines for fully automated builds.

---

## 🧩 Example Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run lineup generation
python src/build_lineups.py

# Run simulations for a sample slate
python src/game_sims.py
```

Results (lineups, ownership summaries, etc.) will be stored in the respective output directories.

---

## 🔍 Technical Highlights

* **OR-Tools (MIP/SAT) optimization** for lineup/field generation
* **Monte Carlo game simulations** for player fantasy point estimation
* **Vectorized ownership decay** for efficient field modeling
* **Stack-based constraint modeling** (5-3, 4-4, 4-3-1, etc.)
* **Hybrid & distributed support** (Lambda-ready architecture)
* **Clean modular design** — each component can be reused independently

---

## 🧠 Extensibility Ideas

* Integrate real projections and live data feeds
* Add post-processing for EV optimization
* Extend methodology to other sports

---

## ⚠️ Disclaimer

This repository is provided **for educational and demonstrative purposes only**.
It omits proprietary logic, real data feeds, and any competitive modeling strategies.
All included examples are placeholders designed to illustrate architecture and software design principles.

**Basically, don't use this and expect to make money.**
