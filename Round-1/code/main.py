import pandas as pd
import os
from data_loader import load_all_raw, filter_t20, build_master
from metrics     import compute_impact_scores
from scarcity    import apply_scarcity
from risk        import compute_risk
from optimizer   import run_optimizer
from visualizer  import generate_all

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("=" * 60)
    print("  RR IPL 2026 — SupeRR Selector Auction Strategy")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────
    print("\n[1/6] Loading raw data...")
    players, auction, bids, comps, matches, stats, mapping = load_all_raw()

    # ── 2. Filter T20 ────────────────────────────────────────
    print("\n[2/6] Filtering T20 matches...")
    t20_matches, stats_t20 = filter_t20(comps, matches, stats)

    # ── 3. Build master ──────────────────────────────────────
    print("\n[3/6] Building master player table...")
    master, _, _ = build_master(players, auction, stats_t20, t20_matches)

    # Exclude retained RR players & require auction history
    candidates = master[
        (~master['is_retained']) &
        (master['base_price_cr'].notna())
    ].copy()
    print(f"  Candidates after filtering: {len(candidates):,}")

    # ── 4. Impact scores ─────────────────────────────────────
    print("\n[4/6] Computing impact scores...")
    candidates = compute_impact_scores(candidates)

    # ── 5. Scarcity + Risk ───────────────────────────────────
    print("\n[5/6] Applying scarcity, value economics & risk...")
    candidates = apply_scarcity(candidates, auction, bids)
    candidates = compute_risk(candidates, stats_t20, t20_matches)

    # ── 6. Optimise ──────────────────────────────────────────
    print("\n[6/6] Running integer optimizer...")
    result = run_optimizer(candidates)

    # ── Save ─────────────────────────────────────────────────
    candidates.to_csv(os.path.join(OUT_DIR, 'all_player_scores.csv'), index=False)
    result.to_csv(os.path.join(OUT_DIR,     'recommended_9.csv'))

    # ── Print summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RECOMMENDED 9 TARGET PLAYERS")
    print("=" * 60)
    display_cols = ['player_name','derived_role','nationality',
                    'max_bid','final_score','risk']
    print(result[[c for c in display_cols if c in result.columns]].to_string())
    total = result['max_bid'].sum()
    print(f"\n  Total Committed : ₹{total:.2f} Cr / ₹{16.05} Cr")
    print(f"  Purse Utilization: {total/16.05*100:.1f}%")

    # ── Charts ───────────────────────────────────────────────
    print("\n[+] Generating visualizations...")
    generate_all(result, auction)
    print("\n✅ Done! See ../outputs/ for all results.")

if __name__ == '__main__':
    main()