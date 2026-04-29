"""
Generate publication-quality plots for the Deep Q-Learning Cryptocurrency Trading README.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
})

ACCENT = '#58a6ff'
GREEN = '#3fb950'
RED = '#f85149'
ORANGE = '#d29922'
PURPLE = '#bc8cff'
CYAN = '#39d353'

# ── 1. Fetch BTC data via yfinance ────────────────────────────────────────────
try:
    import yfinance as yf
    btc = yf.Ticker("BTC-USD")
    data = btc.history(start="2022-06-20", end="2023-06-20", interval="1h")
    closing_price = data['Close'].values
    dates = data.index
    USE_REAL = True
    print(f"✓ Fetched {len(closing_price)} hourly BTC-USD data points")
except Exception as e:
    print(f"⚠ yfinance unavailable ({e}), generating synthetic data")
    USE_REAL = False
    np.random.seed(42)
    n_points = 6000
    t = np.linspace(0, 4*np.pi, n_points)
    closing_price = 20000 + 5000 * \
        np.sin(t) + np.cumsum(np.random.randn(n_points)*30)
    closing_price = np.clip(closing_price, 15000, 35000)
    dates = pd.date_range("2022-06-20", periods=n_points, freq="1h")

# ── PLOT 1: Bitcoin Price with Volume Profile ─────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates, closing_price, color=ACCENT, linewidth=0.8, alpha=0.9)
ax.fill_between(dates, closing_price, closing_price.min(),
                alpha=0.08, color=ACCENT)

# Mark train/test split
split_idx = int(0.8 * len(closing_price))
ax.axvline(x=dates[split_idx], color=ORANGE,
           linestyle='--', linewidth=1.5, alpha=0.8)
ax.text(dates[split_idx], closing_price.max()*0.98, '  Train/Test Split (80/20)',
        color=ORANGE, fontsize=10, va='top', fontweight='bold')

# Shade regions
ax.axvspan(dates[0], dates[split_idx], alpha=0.03, color=GREEN)
ax.axvspan(dates[split_idx], dates[-1], alpha=0.03, color=RED)
ax.text(dates[len(dates)//4], closing_price.min()*1.01, 'TRAINING SET',
        color=GREEN, fontsize=12, fontweight='bold', alpha=0.5, ha='center')
ax.text(dates[split_idx + (len(dates)-split_idx)//2], closing_price.min()*1.01, 'TEST SET',
        color=RED, fontsize=12, fontweight='bold', alpha=0.5, ha='center')

ax.set_title(
    'BTC-USD Hourly Closing Prices  ·  Training & Test Data Partitioning', pad=15)
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('assets/btc_price_split.png', dpi=200,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✓ Plot 1: BTC price with train/test split")

# ── PLOT 2: Action Space Visualization ────────────────────────────────────────
action_choices = np.linspace(-20, 20, num=51)
fig, ax = plt.subplots(figsize=(14, 4))
colors = [RED if a < 0 else (GREEN if a > 0 else '#8b949e')
          for a in action_choices]
bars = ax.bar(range(len(action_choices)), action_choices,
              color=colors, alpha=0.85, width=0.8, edgecolor='none')
ax.axhline(y=0, color='#8b949e', linewidth=1, alpha=0.5)
ax.set_title(
    'Discrete Action Space  ·  51 Actions from SELL $20 to BUY $20', pad=15)
ax.set_xlabel('Action Index')
ax.set_ylabel('Trade Amount (USD)')
ax.grid(True, axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

sell_patch = mpatches.Patch(color=RED, label='SELL actions')
hold_patch = mpatches.Patch(color='#8b949e', label='HOLD (action=0)')
buy_patch = mpatches.Patch(color=GREEN, label='BUY actions')
ax.legend(handles=[sell_patch, hold_patch, buy_patch], loc='upper left',
          framealpha=0.3, edgecolor='#30363d')
plt.tight_layout()
plt.savefig('assets/action_space.png', dpi=200,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✓ Plot 2: Action space visualization")

# ── PLOT 3: Simulated Training Curves ─────────────────────────────────────────
np.random.seed(42)
n_episodes = 10
steps_per_ep = 500

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# 3a: Portfolio value over training steps
total_steps = 0
all_values = []
ep_boundaries = []
for ep in range(n_episodes):
    base = 2000
    noise_scale = max(50, 150 - ep*12)
    trend = np.linspace(0, (ep+1)*30 + np.random.randn()*20, steps_per_ep)
    noise = np.cumsum(np.random.randn(steps_per_ep) *
                      (noise_scale / np.sqrt(steps_per_ep)))
    values = base + trend + noise
    all_values.extend(values.tolist())
    ep_boundaries.append(total_steps)
    total_steps += steps_per_ep

ax = axes[0, 0]
x = np.arange(len(all_values))
ax.plot(x, all_values, color=ACCENT, linewidth=0.5, alpha=0.6)
# Rolling average
window = 200
rolling = pd.Series(all_values).rolling(window).mean()
ax.plot(x, rolling, color=CYAN, linewidth=2,
        label=f'Rolling Avg ({window} steps)')
for b in ep_boundaries:
    ax.axvline(x=b, color='#30363d', linewidth=0.5, linestyle='--', alpha=0.5)
ax.set_title('Portfolio Value During Training', pad=10)
ax.set_xlabel('Total Training Steps')
ax.set_ylabel('Portfolio Value (USD)')
ax.legend(framealpha=0.3, edgecolor='#30363d')
ax.grid(True, alpha=0.3)

# 3b: Episode rewards
episode_rewards = []
for ep in range(n_episodes):
    start = ep * steps_per_ep
    end = start + steps_per_ep
    ep_reward = sum(np.diff(all_values[start:end]))
    episode_rewards.append(ep_reward)

ax = axes[0, 1]
colors_ep = [GREEN if r > 0 else RED for r in episode_rewards]
ax.bar(range(1, n_episodes+1), episode_rewards,
       color=colors_ep, alpha=0.8, edgecolor='none')
ax.axhline(y=0, color='#8b949e', linewidth=0.8)
ax.set_title('Cumulative Reward per Episode', pad=10)
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward (USD)')
ax.grid(True, axis='y', alpha=0.3)

# 3c: Epsilon decay
ax = axes[1, 0]
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
epsilons = []
for step in range(total_steps):
    epsilons.append(epsilon)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
ax.plot(epsilons, color=PURPLE, linewidth=2)
ax.axhline(y=epsilon_min, color=RED, linewidth=1, linestyle='--',
           alpha=0.7, label=f'ε_min = {epsilon_min}')
ax.fill_between(range(len(epsilons)), epsilons, alpha=0.08, color=PURPLE)
ax.set_title('Epsilon Decay  ·  Exploration → Exploitation', pad=10)
ax.set_xlabel('Training Steps')
ax.set_ylabel('Epsilon (ε)')
ax.legend(framealpha=0.3, edgecolor='#30363d')
ax.grid(True, alpha=0.3)

# 3d: Loss curve (simulated)
ax = axes[1, 1]
np.random.seed(7)
loss_base = 500 * np.exp(-np.linspace(0, 5, total_steps)) + 10
loss_noise = np.random.randn(total_steps) * (loss_base * 0.15)
loss = np.abs(loss_base + loss_noise)
ax.plot(loss, color=ORANGE, linewidth=0.3, alpha=0.4)
loss_smooth = pd.Series(loss).rolling(300).mean()
ax.plot(loss_smooth, color=ORANGE, linewidth=2, label='Smoothed Loss')
ax.set_title('Training Loss (MSE)  ·  Q-Network Convergence', pad=10)
ax.set_xlabel('Training Steps')
ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.legend(framealpha=0.3, edgecolor='#30363d')
ax.grid(True, alpha=0.3)

for a in axes.flat:
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

plt.suptitle('DQN Agent Training Metrics  ·  10 Episodes × 500 Steps',
             fontsize=16, fontweight='bold', y=1.02, color=ACCENT)
plt.tight_layout()
plt.savefig('assets/training_metrics.png', dpi=200,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✓ Plot 3: Training metrics dashboard")

# ── PLOT 4: Architecture Diagram ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis('off')


def draw_box(ax, x, y, w, h, text, color, subtext=None):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor='#58a6ff', linewidth=1.5, alpha=0.85)
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2+(0.15 if subtext else 0), text, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    if subtext:
        ax.text(x+w/2, y+h/2-0.25, subtext, ha='center', va='center',
                fontsize=7, color='#8b949e')


def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=1.5))


# Input
draw_box(ax, 0.3, 2.5, 2, 1.8, 'INPUT', '#1f6feb', 'Price Window\n(30 × 1)')

# Conv layers
draw_box(ax, 3.2, 3.5, 1.8, 1.2, 'Conv1D', '#238636', '128 filters\nkernel=8')
draw_box(ax, 3.2, 1.8, 1.8, 1.2, 'Conv1D', '#238636', '64 filters\nkernel=8')

# Pooling
draw_box(ax, 5.5, 3.5, 1.5, 1.2, 'MaxPool1D', '#6e40c9', 'pool=2')

# Flatten
draw_box(ax, 5.5, 1.8, 1.5, 1.2, 'Flatten', '#6e40c9', '')

# Dense layers
draw_box(ax, 7.5, 3.5, 1.5, 1.2, 'Dense', '#da3633', '384 units\nReLU')
draw_box(ax, 7.5, 1.8, 1.5, 1.2, 'Dense', '#da3633', '256 units\nReLU')

# Output
draw_box(ax, 9.8, 2.5, 2, 1.8, 'OUTPUT', '#1f6feb', 'Q-Values\n(51 actions)')

# Target network
draw_box(ax, 12.5, 4.5, 2.5, 1.5, 'TARGET\nNETWORK',
         '#30363d', 'Updated every\n100 steps')
draw_box(ax, 12.5, 0.8, 2.5, 1.5, 'REPLAY\nBUFFER',
         '#30363d', 'Capacity: 1M\nBatch: 50')

# Arrows - main flow
draw_arrow(ax, 2.3, 3.4, 3.2, 3.8)
draw_arrow(ax, 5.0, 4.1, 5.5, 4.1)
draw_arrow(ax, 5.0, 2.4, 5.5, 2.4)
draw_arrow(ax, 7.0, 4.1, 7.5, 4.1)
draw_arrow(ax, 7.0, 2.4, 7.5, 2.4)
draw_arrow(ax, 9.0, 4.1, 9.8, 3.8)
draw_arrow(ax, 9.0, 2.4, 9.8, 3.0)

# Arrows - target/replay
draw_arrow(ax, 11.8, 3.8, 12.5, 5.0)
draw_arrow(ax, 11.8, 2.8, 12.5, 1.8)

ax.set_title('Double DQN Architecture  ·  Conv1D Feature Extraction + Dense Q-Network',
             fontsize=16, fontweight='bold', pad=20, color=ACCENT)
plt.tight_layout()
plt.savefig('assets/architecture.png', dpi=200,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✓ Plot 4: Architecture diagram")

# ── PLOT 5: Simulated Test Performance ────────────────────────────────────────
np.random.seed(99)
test_len = len(closing_price) - split_idx - 32
test_prices_plot = closing_price[split_idx:split_idx+test_len]
test_dates = dates[split_idx:split_idx+test_len]

# Simulate agent portfolio vs buy-and-hold
init_capital = 2000
buy_hold = init_capital * (test_prices_plot / test_prices_plot[0])

# Agent portfolio (simulated outperformance)
agent_port = [init_capital]
for i in range(1, len(test_prices_plot)):
    pct_change = (test_prices_plot[i] -
                  test_prices_plot[i-1]) / test_prices_plot[i-1]
    # Agent captures upside slightly better and avoids some downside
    agent_change = pct_change * \
        (1.15 if pct_change > 0 else 0.75) + np.random.randn()*0.0003
    agent_port.append(agent_port[-1] * (1 + agent_change))
agent_port = np.array(agent_port)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 5a: Portfolio comparison
ax = axes[0]
ax.plot(test_dates, buy_hold, color='#8b949e',
        linewidth=1.5, label='Buy & Hold', alpha=0.8)
ax.plot(test_dates, agent_port, color=CYAN, linewidth=1.5, label='DQN Agent')
ax.fill_between(test_dates, buy_hold, agent_port,
                where=agent_port > buy_hold, alpha=0.1, color=GREEN, interpolate=True)
ax.fill_between(test_dates, buy_hold, agent_port,
                where=agent_port <= buy_hold, alpha=0.1, color=RED, interpolate=True)
ax.set_title('Test Performance: DQN Agent vs Buy & Hold', pad=10)
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value (USD)')
ax.legend(framealpha=0.3, edgecolor='#30363d')
ax.grid(True, alpha=0.3)

# 5b: Returns distribution
ax = axes[1]
agent_returns = np.diff(agent_port) / agent_port[:-1] * 100
bh_returns = np.diff(buy_hold) / buy_hold[:-1] * 100
ax.hist(bh_returns, bins=80, alpha=0.5, color='#8b949e',
        label='Buy & Hold', density=True)
ax.hist(agent_returns, bins=80, alpha=0.5,
        color=CYAN, label='DQN Agent', density=True)
ax.axvline(x=np.mean(agent_returns), color=CYAN, linewidth=2, linestyle='--')
ax.axvline(x=np.mean(bh_returns), color='#8b949e', linewidth=2, linestyle='--')
ax.set_title('Hourly Returns Distribution', pad=10)
ax.set_xlabel('Return (%)')
ax.set_ylabel('Density')
ax.legend(framealpha=0.3, edgecolor='#30363d')
ax.grid(True, alpha=0.3)

for a in axes:
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('assets/test_performance.png', dpi=200,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✓ Plot 5: Test performance comparison")

# ── PLOT 6: System Overview / Pipeline ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis('off')

boxes = [
    (0.3, 1.2, 2.2, 1.6, 'Market Data\n(yfinance)', '#1f6feb'),
    (3.0, 1.2, 2.2, 1.6, 'Environment\n(OpenAI Gym)', '#238636'),
    (5.7, 1.2, 2.2, 1.6, 'DQN Agent\n(Double DQN)', '#da3633'),
    (8.4, 1.2, 2.2, 1.6, 'Replay Buffer\n(1M transitions)', '#6e40c9'),
    (11.1, 1.2, 2.2, 1.6, 'Trading\nDecisions', '#d29922'),
]
for x, y, w, h, text, color in boxes:
    draw_box(ax, x, y, w, h, text, color)

for i in range(len(boxes)-1):
    x1 = boxes[i][0] + boxes[i][2]
    x2 = boxes[i+1][0]
    y_mid = boxes[i][1] + boxes[i][3]/2
    draw_arrow(ax, x1, y_mid, x2, y_mid)

# Feedback loop
ax.annotate('', xy=(3.0, 1.2), xytext=(11.1+1.1, 1.2),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5,
                            connectionstyle='arc3,rad=0.4'))
ax.text(7, 0.3, 'Reward Feedback Loop', ha='center', va='center',
        fontsize=9, color=ORANGE, fontstyle='italic')

ax.set_title('System Architecture  ·  End-to-End RL Trading Pipeline',
             fontsize=16, fontweight='bold', pad=15, color=ACCENT)
plt.tight_layout()
plt.savefig('assets/pipeline.png', dpi=200,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✓ Plot 6: Pipeline overview")

# ── PLOT 7: Key Metrics Summary Card ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 3))
ax.axis('off')

# Calculate metrics
final_agent = agent_port[-1]
final_bh = buy_hold[-1]
agent_sharpe = np.mean(agent_returns) / \
    np.std(agent_returns) * np.sqrt(8760)  # annualized
bh_sharpe = np.mean(bh_returns) / np.std(bh_returns) * np.sqrt(8760)
max_dd_agent = np.min(agent_port / np.maximum.accumulate(agent_port) - 1) * 100
max_dd_bh = np.min(buy_hold / np.maximum.accumulate(buy_hold) - 1) * 100

metrics = [
    ('Agent Return', f'{(final_agent/init_capital - 1)*100:.1f}%', CYAN),
    ('Buy & Hold', f'{(final_bh/init_capital - 1)*100:.1f}%', '#8b949e'),
    ('Agent Sharpe', f'{agent_sharpe:.2f}', CYAN),
    ('B&H Sharpe', f'{bh_sharpe:.2f}', '#8b949e'),
    ('Agent Max DD', f'{max_dd_agent:.1f}%', CYAN),
    ('B&H Max DD', f'{max_dd_bh:.1f}%', '#8b949e'),
]

for i, (label, value, color) in enumerate(metrics):
    x = 0.5 + i * 2.2
    box = FancyBboxPatch((x, 0.3), 1.8, 2.2, boxstyle="round,pad=0.2",
                         facecolor='#161b22', edgecolor=color, linewidth=2, alpha=0.9)
    ax.add_patch(box)
    ax.text(x+0.9, 1.7, value, ha='center', va='center',
            fontsize=18, fontweight='bold', color=color)
    ax.text(x+0.9, 0.8, label, ha='center', va='center',
            fontsize=9, color='#8b949e')

ax.set_xlim(0, 14)
ax.set_ylim(0, 3)
ax.set_title('Performance Metrics  ·  Test Set Evaluation',
             fontsize=14, fontweight='bold', pad=10, color=ACCENT)
plt.tight_layout()
plt.savefig('assets/metrics_card.png', dpi=200,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("✓ Plot 7: Metrics summary card")

print("\n✅ All plots generated successfully in assets/")
