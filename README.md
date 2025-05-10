# Smart Order Router Backtest: Cont & Kukanov Model

## Overview

This repository implements a Smart Order Router (SOR) that follows the static cost model introduced by Rama Cont and Arseniy Kukanov in *"Optimal Order Placement in Limit Order Markets"*. The purpose is to back-test and tune the SOR using mocked market data and compare it against baseline execution strategies such as Best Ask, TWAP, and VWAP.

The router aims to execute a 5,000-share buy order optimally across multiple venues by minimizing total cost, factoring in:

- **lambda_over**: penalty for overfilling
- **lambda_under**: penalty for underfilling
- **theta_queue**: penalty related to queue position risk

## Code Structure

- `Venue`: Class representing a trading venue with ask price, size, fee, and rebate.
- `compute_cost`: Computes execution cost with penalties based on fill performance.
- `allocate`: Implements static allocation based on a brute-force grid (step size: 100).
- `twap_strategy`, `vwap_strategy`, `best_ask_strategy`: Baseline strategies for benchmarking.
- `run_backtest`: Main function that:
  - Reconstructs venue snapshots
  - Searches parameter combinations
  - Executes the allocator
  - Evaluates performance vs. baselines
- `generate_param_grid`: Defines a grid over the 3 risk parameters.

## Parameter Search

The grid search explores 27 combinations:

- `lambda_over`: [0.01, 0.05, 0.1]
- `lambda_under`: [0.01, 0.05, 0.1]
- `theta_queue`: [0.0001, 0.0005, 0.001]

The combination that fully executes the order with the lowest cash cost is selected as optimal. If no full execution is possible, the fallback parameters are used.

## Output

After execution, the script prints a structured JSON object with:

- Best parameter set found
- Total cash spent and average fill price
- Baseline statistics (Best Ask, TWAP, VWAP)
- Savings vs. each baseline in basis points

This output can easily be modified to be passed on to other functions.
Uncomment lines #3 and #198 to be able to use the json object.

## Suggested Improvement

**Queue Dynamics Simulation**: Currently, execution assumes immediate fill up to the displayed size. A realistic enhancement would simulate **slippage and queue priority**, for example:
- Introduce a probability-based fill model where deeper sizes or aggressive prices face execution uncertainty.
- Incorporate real-world latency/queue positioning using exponential decay based on time-in-queue or order priority.

Such improvements would bring the model closer to real trading environments and offer deeper insights into optimal order routing.

## Requirements

- Python 3.8+
- `numpy`, `pandas`

No other libraries are used, in accordance with task constraints.

## How to Run

```bash
python3 backtest.py
