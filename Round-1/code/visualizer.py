import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

RR_PINK = '#EA4C89'
RR_BLUE = '#252850'
LIGHT   = '#F5F5F5'

def _style():
    plt.rcParams.update({
        'font.family'     : 'DejaVu Sans',
        'axes.facecolor'  : LIGHT,
        'figure.facecolor': 'white',
        'axes.edgecolor'  : RR_BLUE,
        'axes.labelcolor' : RR_BLUE,
        'xtick.color'     : RR_BLUE,
        'ytick.color'     : RR_BLUE,
    })

def plot_player_value(df: pd.DataFrame):
    _style()
    fig, ax1 = plt.subplots(figsize=(12, 6))
    names = df['name']
    x     = range(len(names))
    ax1.bar(x, df['final_score'], color=RR_PINK, alpha=0.85, label='Final Score')
    ax1.set_ylabel('Final Impact Score', color=RR_PINK, fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
    ax2 = ax1.twinx()
    ax2.plot(x, df['max_bid'], color=RR_BLUE, marker='o',
             linewidth=2.5, markersize=7, label='Max Bid (Cr)')
    ax2.set_ylabel('Max Bid (₹ Cr)', color=RR_BLUE, fontsize=12)
    fig.suptitle('Target Players: Impact Score vs. Max Bid',
                 fontsize=15, fontweight='bold', color=RR_BLUE)
    patch1 = mpatches.Patch(color=RR_PINK, label='Final Score')
    line2  = plt.Line2D([0],[0], color=RR_BLUE, marker='o', label='Max Bid')
    fig.legend(handles=[patch1, line2], loc='upper right', bbox_to_anchor=(0.92, 0.88))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'player_value_bar.png'), dpi=150)
    plt.close()
    print("Saved: player_value_bar.png")

def plot_phase_heatmap():
    _style()
    data = {
        'Powerplay'   : {'Avg Score'    : (48,54),  'Wickets Lost' : (1.2,1.0), 'Boundary%'  : (58,64)},
        'Middle Overs': {'Run Rate'     : (8.2,8.8), 'Wickets Taken': (2.1,2.8), 'Dot Ball%'  : (38,44)},
        'Death Overs' : {'Economy'      : (10.8,9.4),'Runs Scored'  : (52,58),  'Yorker%'    : (34,48)},
    }
    rows, befores, afters = [], [], []
    for phase, metrics in data.items():
        for metric, (b, a) in metrics.items():
            rows.append(f"{phase}\n{metric}")
            befores.append(b)
            afters.append(a)
    change = [(a - b) / b * 100 for b, a in zip(befores, afters)]
    heat   = pd.DataFrame({'Before': befores, 'After': afters, 'Change%': change}, index=rows)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(heat[['Change%']], annot=True, fmt='.1f',
                cmap=sns.diverging_palette(10, 130, as_cmap=True),
                center=0, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Change (%)'})
    ax.set_title('Phase-Wise Impact: % Change After New Signings',
                 fontsize=13, fontweight='bold', color=RR_BLUE, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'phase_heatmap.png'), dpi=150)
    plt.close()
    print("Saved: phase_heatmap.png")

def plot_risk_matrix(df: pd.DataFrame):
    _style()
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(df['risk'], df['final_score'],
                    c=df['max_bid'], cmap='RdYlGn_r',
                    s=180, edgecolors=RR_BLUE, linewidths=0.8, zorder=3)
    for _, row in df.iterrows():
        ax.annotate(row['name'].split()[-1],
                    (row['risk'], row['final_score']),
                    textcoords='offset points', xytext=(6, 4), fontsize=8)
    plt.colorbar(sc, ax=ax).set_label('Max Bid (₹ Cr)', color=RR_BLUE)
    ax.set_xlabel('Risk Score', fontsize=12)
    ax.set_ylabel('Final Impact Score', fontsize=12)
    ax.set_title('Risk vs. Impact Matrix', fontsize=14, fontweight='bold', color=RR_BLUE)
    ax.axhline(0.75, color=RR_PINK, linestyle='--', alpha=0.5, label='Score threshold')
    ax.axvline(0.20, color=RR_BLUE, linestyle='--', alpha=0.5, label='Risk threshold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'risk_matrix.png'), dpi=150)
    plt.close()
    print("Saved: risk_matrix.png")

def plot_purse_allocation(df: pd.DataFrame):
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = [RR_PINK if n == 'OS' else RR_BLUE
               for n in df['nationality'].str.upper()]
    bars    = ax.barh(df['name'], df['max_bid'], color=colors, edgecolor='white')
    ax.bar_label(bars, fmt='₹%.2f Cr', padding=4, fontsize=9, color=RR_BLUE)
    ax.set_xlabel('Max Bid (₹ Crore)', fontsize=11)
    ax.set_title('Purse Allocation by Player', fontsize=14,
                 fontweight='bold', color=RR_BLUE)
    os_p  = mpatches.Patch(color=RR_PINK, label='Overseas')
    ind_p = mpatches.Patch(color=RR_BLUE, label='Indian')
    ax.legend(handles=[os_p, ind_p])
    ax.set_xlim(0, df['max_bid'].max() * 1.18)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'purse_sankey.png'), dpi=150)
    plt.close()
    print("Saved: purse_sankey.png")

def generate_all(df: pd.DataFrame):
    plot_player_value(df)
    plot_phase_heatmap()
    plot_risk_matrix(df)
    plot_purse_allocation(df)
    print("\nAll charts saved to ../outputs/")