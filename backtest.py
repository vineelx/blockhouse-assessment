import pandas as pd
import numpy as np
#import json

class Venue:
    def __init__(self, ask, ask_size, fee=0.0002, rebate=0.0001):
        self.ask = ask
        self.ask_size = ask_size
        self.fee = fee
        self.rebate = rebate

def compute_cost(split, venues, order_size, lambda_over, lambda_under, theta_queue):
    executed = 0
    cash_spent = 0
    for i, venue in enumerate(venues):
        exe = min(split[i], venue.ask_size)
        executed += exe
        cash_spent += exe * (venue.ask + venue.fee)
        maker_rebate = max(split[i] - exe, 0) * venue.rebate
        cash_spent -= maker_rebate
    
    underfill = max(order_size - executed, 0)
    overfill = max(executed - order_size, 0)
    risk_pen = theta_queue * (underfill + overfill)
    cost_pen = lambda_under * underfill + lambda_over * overfill
    return cash_spent + risk_pen + cost_pen


def allocate(order_size, venues, lambda_over, lambda_under, theta_queue):
    step = 100
    splits = [[]]
    
    for v in range(len(venues)):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size - used, venues[v].ask_size)
            for q in range(0, max_v + 1, step):
                new_splits.append(alloc + [q])
        splits = new_splits

    best_cost = float("inf")
    best_split = None
    for alloc in splits:
        if sum(alloc) != order_size:
            continue
        cost = compute_cost(alloc, venues, order_size, lambda_over, lambda_under, theta_queue)
        if cost < best_cost:
            best_cost = cost
            best_split = alloc
    return best_split, best_cost

def twap_strategy(venue_snapshots, order_size):
    if not venue_snapshots:
        return 0, 0
    
    chunks = len(venue_snapshots)
    chunk_size = order_size // chunks
    remainder = order_size % chunks
    
    executed = 0
    cash = 0
    for i, venues in enumerate(venue_snapshots):
        current_chunk = chunk_size + (1 if i < remainder else 0)
        if current_chunk <= 0:
            continue
            
        best_venue = min(venues, key=lambda v: v.ask)
        filled = min(current_chunk, best_venue.ask_size)
        executed += filled
        cash += filled * (best_venue.ask + best_venue.fee)
    
    return cash, cash / executed if executed > 0 else 0

def vwap_strategy(venue_snapshots, order_size):
    executed = 0
    cash = 0
    remaining = order_size
    
    for venues in venue_snapshots:
        if remaining <= 0:
            break
            
        total_size = sum(v.ask_size for v in venues)
        if total_size == 0:
            continue
            
        for v in venues:
            w = v.ask_size / total_size
            alloc = min(int(order_size * w), v.ask_size, remaining)
            executed += alloc
            cash += alloc * (v.ask + v.fee)
            remaining -= alloc
            if remaining <= 0:
                break
                
    return cash, cash / executed if executed > 0 else 0

def best_ask_strategy(venue_snapshots, order_size):
    executed = 0
    cash = 0
    remaining = order_size
    
    for venues in venue_snapshots:
        if remaining <= 0:
            break
            
        best = sorted(venues, key=lambda v: v.ask)
        for v in best:
            filled = min(v.ask_size, remaining)
            cash += filled * (v.ask + v.fee)
            executed += filled
            remaining -= filled
            if remaining <= 0:
                break
                
    return cash, cash / executed if executed > 0 else 0

def run_backtest(df, param_grid, order_size):
    # Create venue snapshots
    snapshots = []
    for ts, group in df.groupby("ts_event"):
        venues = [Venue(r.ask_px_00, r.ask_sz_00) for _, r in group.iterrows()]
        snapshots.append(venues)
    
    # Evaluate all parameter combinations
    best_params = None
    best_cash = float("inf")
    best_avg_price = None
    best_filled = 0
    
    for lamb_o, lamb_u, theta in param_grid:
        remaining = order_size
        cash = 0
        filled = 0
        
        for venues in snapshots:
            if remaining <= 0:
                break
                
            qty = min(remaining, order_size)
            split, _ = allocate(qty, venues, lamb_o, lamb_u, theta)
            
            if split is None:
                continue  # No valid allocation for this snapshot
                
            for i, v in enumerate(venues):
                executed = min(split[i], v.ask_size)
                cash += executed * (v.ask + v.fee)
                remaining -= executed
                filled += executed
                if remaining <= 0:
                    break
        
        # Only consider parameter sets that filled the entire order
        if remaining == 0 and cash < best_cash:
            best_cash = cash
            best_avg_price = cash / order_size
            best_params = (lamb_o, lamb_u, theta)
            best_filled = filled
    
    # If no parameters filled the entire order, use the one that filled the most
    if best_params is None:
        best_params = (0.01, 0.01, 0.0001)  # Default fallback
        best_avg_price = 0
        best_cash = 0
    
    # Calculate baselines
    twap_cash, twap_avg = twap_strategy(snapshots, order_size)
    vwap_cash, vwap_avg = vwap_strategy(snapshots, order_size)
    ba_cash, ba_avg = best_ask_strategy(snapshots, order_size)
    
    # Calculate savings in basis points
    def calc_savings(baseline_avg, our_avg):
        if baseline_avg == 0 or our_avg == 0:
            return None
        return round(1e4 * (baseline_avg - our_avg) / baseline_avg, 4)
    
    output = {
        "best_parameters": {
            "lambda_over": best_params[0],
            "lambda_under": best_params[1],
            "theta_queue": best_params[2]
        },
        "best_total_cash": round(best_cash, 2),
        "best_average_fill_price": round(best_avg_price, 6),
        "baselines": {
            "best_ask": {"cash": round(ba_cash, 2), "avg_price": round(ba_avg, 6)},
            "twap": {"cash": round(twap_cash, 2), "avg_price": round(twap_avg, 6)},
            "vwap": {"cash": round(vwap_cash, 2), "avg_price": round(vwap_avg, 6)}
        },
        "savings_vs_baselines_bps": {
            "best_ask": calc_savings(ba_avg, best_avg_price),
            "twap": calc_savings(twap_avg, best_avg_price),
            "vwap": calc_savings(vwap_avg, best_avg_price)
        }
    }
    #print(json.dumps(output, indent=2))
    print("{")
    print('  "best_parameters": {')
    print(f'    "lambda_over": {best_params[0]},')
    print(f'    "lambda_under": {best_params[1]},')
    print(f'    "theta_queue": {best_params[2]}')
    print("  },")
    print(f'  "best_total_cash": {round(best_cash, 2)},')
    print(f'  "best_average_fill_price": {round(best_avg_price, 6)},')
    print('  "baselines": {')
    print('    "best_ask": {')
    print(f'      "cash": {round(ba_cash, 2)},')
    print(f'      "avg_price": {round(ba_avg, 6)}')
    print("    },")
    print('    "twap": {')
    print(f'      "cash": {round(twap_cash, 2)},')
    print(f'      "avg_price": {round(twap_avg, 6)}')
    print("    },")
    print('    "vwap": {')
    print(f'      "cash": {round(vwap_cash, 2)},')
    print(f'      "avg_price": {round(vwap_avg, 6)}')
    print("    }")
    print("  },")
    print('  "savings_vs_baselines_bps": {')
    print(f'    "best_ask": {calc_savings(ba_avg, best_avg_price)},')
    print(f'    "twap": {calc_savings(twap_avg, best_avg_price)},')
    print(f'    "vwap": {calc_savings(vwap_avg, best_avg_price)}')
    print("  }")
    print("}")

def generate_param_grid():
    lambda_overs = [0.01, 0.05, 0.1]
    lambda_unders = [0.01, 0.05, 0.1]
    theta_queues = [0.0001, 0.0005, 0.001]
    
    param_grid = []
    for lo in lambda_overs:
        for lu in lambda_unders:
            for tq in theta_queues:
                param_grid.append((lo, lu, tq))
    return param_grid

df = pd.read_csv("l1_day.csv")
df = df.drop_duplicates(subset=["ts_event", "publisher_id"])
df["ts_event"] = pd.to_datetime(df["ts_event"])
df.sort_values("ts_event", inplace=True)

# Grid search over parameter values
param_grid = generate_param_grid()
order_size = 5000

run_backtest(df, param_grid, order_size)