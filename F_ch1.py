"""
bandit_app.py — An educational, interactive implementation of Chapter 2 (Multi-armed Bandits)
from Sutton & Barto (RL book).

What you get:
1) Core objects:
   - Bandit (environment): hidden true values q*(a), noisy rewards
   - Agent (learner): Q estimates Q_t(a), counts N_t(a)

2) Strategies (selectable):
   - ε-greedy (exploration vs exploitation)
   - Optimistic initial values (greedy with Q1(a)=+5, ε=0)
   - UCB (Upper Confidence Bound)

3) Two learning step-size modes:
   - Sample-average (α = 1 / N(a))  [stationary]
   - Constant step-size (α = constant) [non-stationary friendly]

4) UI modes:
   - Manual vs AI: you click arms; AI shows what it would pick & why; update math shown step-by-step
   - Simulation run: replicate the book-style plots (Average Reward, % Optimal Action) and compare methods

Run:
    pip install streamlit numpy pandas matplotlib
    streamlit run bandit_app.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Helper: argmax with random tie-breaking
# -----------------------------
def argmax_random_tie(values: np.ndarray, rng: np.random.Generator) -> int:
    """Return an index of the maximum value, breaking ties uniformly at random."""
    max_val = np.max(values)
    candidates = np.flatnonzero(values == max_val)
    return int(rng.choice(candidates))


# -----------------------------
# Bandit (Environment)
# -----------------------------
@dataclass
class GaussianBandit:
    """
    10-armed testbed-style bandit:
      q*(a) ~ Normal(0, 1)
      reward R ~ Normal(q*(a), 1)
    """
    k: int
    rng: np.random.Generator
    q_true: np.ndarray

    @classmethod
    def new(cls, k: int, rng: np.random.Generator) -> "GaussianBandit":
        q_true = rng.normal(loc=0.0, scale=1.0, size=k)  # q*(a)
        return cls(k=k, rng=rng, q_true=q_true)

    def optimal_action(self) -> int:
        return int(np.argmax(self.q_true))

    def reward(self, action: int) -> float:
        # noisy reward: Normal(mean=q*(action), var=1)
        return float(self.rng.normal(loc=self.q_true[action], scale=1.0))


@dataclass
class RandomWalkGaussianBandit(GaussianBandit):
    """
    Non-stationary bandit variant (common exercise):
    After each step, each q*(a) performs an independent random walk.
    """
    walk_sigma: float = 0.01

    @classmethod
    def new(cls, k: int, rng: np.random.Generator, walk_sigma: float = 0.01) -> "RandomWalkGaussianBandit":
        q_true = rng.normal(loc=0.0, scale=1.0, size=k)
        return cls(k=k, rng=rng, q_true=q_true, walk_sigma=walk_sigma)

    def drift(self) -> None:
        self.q_true += self.rng.normal(loc=0.0, scale=self.walk_sigma, size=self.k)


# -----------------------------
# Agent (Learner)
# -----------------------------
StepMode = Literal["sample_average", "constant_alpha"]
Strategy = Literal["epsilon_greedy", "optimistic_greedy", "ucb"]


@dataclass
class AgentConfig:
    k: int = 10
    strategy: Strategy = "epsilon_greedy"
    step_mode: StepMode = "sample_average"   # sample_average or constant_alpha
    epsilon: float = 0.1                     # used by epsilon_greedy
    alpha: float = 0.1                       # used by constant_alpha updates
    c: float = 2.0                           # used by UCB
    initial_q: float = 0.0                   # 0 for realistic, +5 for optimistic


class BanditAgent:
    """
    Stores:
      - Q estimates: Q[a]
      - counts: N[a]
    Provides:
      - choose_action(t): based on selected strategy
      - update(action, reward): incremental update
    """

    def __init__(self, cfg: AgentConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.Q = np.full(cfg.k, cfg.initial_q, dtype=float)
        self.N = np.zeros(cfg.k, dtype=int)

    def reset(self) -> None:
        self.Q[:] = self.cfg.initial_q
        self.N[:] = 0

    def _alpha_for(self, action: int) -> float:
        """
        Step size α:
          - sample_average: α = 1 / N(action)  (after incrementing N)
          - constant_alpha: α = cfg.alpha
        """
        if self.cfg.step_mode == "sample_average":
            # N(action) must be >= 1 here
            return 1.0 / float(self.N[action])
        return float(self.cfg.alpha)

    def choose_action(self, t: int) -> Tuple[int, Dict]:
        """
        Choose an action A_t and also return an 'explanation dict' for teaching.
        t is 1-indexed time step in the simulation.
        """
        cfg = self.cfg

        if cfg.strategy == "optimistic_greedy":
            # optimistic initial values typically used with greedy (ε=0)
            greedy_action = argmax_random_tie(self.Q, self.rng)
            info = {
                "strategy": "Optimistic Greedy",
                "why": "Exploit (greedy). Because Q was initialized optimistically, greedy still explores early.",
                "picked": greedy_action,
                "Q_snapshot": self.Q.copy(),
            }
            return greedy_action, info

        if cfg.strategy == "epsilon_greedy":
            u = float(self.rng.random())
            if u < cfg.epsilon:
                a = int(self.rng.integers(0, cfg.k))
                info = {
                    "strategy": "ε-Greedy",
                    "mode": "Explore",
                    "why": f"Random number u={u:.4f} < ε={cfg.epsilon:.4f} → choose a random arm.",
                    "picked": a,
                }
                return a, info
            else:
                a = argmax_random_tie(self.Q, self.rng)
                info = {
                    "strategy": "ε-Greedy",
                    "mode": "Exploit",
                    "why": f"Random number u={u:.4f} ≥ ε={cfg.epsilon:.4f} → choose greedy argmax Q.",
                    "picked": a,
                    "Q_snapshot": self.Q.copy(),
                }
                return a, info

        if cfg.strategy == "ucb":
            # UCB: choose argmax_a [ Q(a) + c * sqrt( ln(t) / N(a) ) ]
            # If N(a)=0, treat as "maximizing" (try each arm at least once).
            untried = np.flatnonzero(self.N == 0)
            if len(untried) > 0:
                a = int(self.rng.choice(untried))
                info = {
                    "strategy": "UCB",
                    "mode": "Forced initial exploration",
                    "why": f"N(a)=0 for some arms, so pick one untried arm (book rule).",
                    "picked": a,
                    "untried_arms": untried.tolist(),
                }
                return a, info

            bonus = cfg.c * np.sqrt(np.log(max(t, 2)) / self.N.astype(float))
            ucb_score = self.Q + bonus
            a = argmax_random_tie(ucb_score, self.rng)
            info = {
                "strategy": "UCB",
                "mode": "UCB maximize",
                "why": "Pick arm that maximizes Q(a) + exploration bonus.",
                "picked": a,
                "Q_snapshot": self.Q.copy(),
                "N_snapshot": self.N.copy(),
                "bonus": bonus.copy(),
                "ucb_score": ucb_score.copy(),
            }
            return a, info

        raise ValueError(f"Unknown strategy: {cfg.strategy}")

    def update(self, action: int, reward: float) -> Dict:
        """
        Incremental update:
            Q <- Q + α [R - Q]
        Where α is either 1/N(action) (sample-average) or constant α.

        Returns a dict with the math for educational display.
        """
        old_Q = float(self.Q[action])

        # increment count first (so sample-average uses the new N)
        self.N[action] += 1
        alpha = self._alpha_for(action)

        error = reward - old_Q
        new_Q = old_Q + alpha * error
        self.Q[action] = new_Q

        return {
            "action": action,
            "reward": reward,
            "old_Q": old_Q,
            "N_after": int(self.N[action]),
            "alpha": float(alpha),
            "error": float(error),
            "new_Q": float(new_Q),
            "update_formula": "Q <- Q + α(R - Q)",
        }


# -----------------------------
# Simulation (Figures-style)
# -----------------------------
@dataclass
class SimResult:
    avg_reward: np.ndarray        # shape [steps]
    pct_optimal: np.ndarray       # shape [steps]
    label: str


def run_one(method_cfg: AgentConfig, steps: int, rng: np.random.Generator, nonstationary: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Run a single bandit problem for 'steps' time steps."""
    if nonstationary:
        bandit = RandomWalkGaussianBandit.new(method_cfg.k, rng)
    else:
        bandit = GaussianBandit.new(method_cfg.k, rng)

    agent = BanditAgent(method_cfg, rng)

    rewards = np.zeros(steps, dtype=float)
    optimal_hits = np.zeros(steps, dtype=float)
    opt_a = bandit.optimal_action()

    for t in range(1, steps + 1):
        a, _info = agent.choose_action(t)
        r = bandit.reward(a)
        agent.update(a, r)

        rewards[t - 1] = r
        optimal_hits[t - 1] = 1.0 if a == opt_a else 0.0

        if nonstationary and isinstance(bandit, RandomWalkGaussianBandit):
            bandit.drift()
            opt_a = bandit.optimal_action()  # optimal action can change over time

    return rewards, optimal_hits


@st.cache_data(show_spinner=False)
def run_many(method_cfg: AgentConfig, steps: int, runs: int, seed: int, nonstationary: bool, label: str) -> SimResult:
    """Average over many independent runs (each run has a new bandit problem)."""
    rng_master = np.random.default_rng(seed)

    sum_rewards = np.zeros(steps, dtype=float)
    sum_opt = np.zeros(steps, dtype=float)

    for _ in range(runs):
        # independent RNG per run
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        rewards, optimal_hits = run_one(method_cfg, steps, rng, nonstationary)
        sum_rewards += rewards
        sum_opt += optimal_hits

    avg_reward = sum_rewards / float(runs)
    pct_optimal = 100.0 * (sum_opt / float(runs))
    return SimResult(avg_reward=avg_reward, pct_optimal=pct_optimal, label=label)


def plot_curves(results: List[SimResult], title: str, y_label: str) -> plt.Figure:
    fig = plt.figure()
    for res in results:
        plt.plot(res.avg_reward if y_label == "Average reward" else res.pct_optimal, label=res.label)
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.25)
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
def ui_sidebar() -> Dict:
    st.sidebar.title("Bandit Lab (Chapter 2)")

    mode = st.sidebar.radio(
        "Choose mode",
        ["Manual vs AI (Interactive)", "Simulation Run (Plots)"],
        index=0
    )

    st.sidebar.subheader("Bandit settings")
    k = st.sidebar.number_input("Number of arms (k)", min_value=2, max_value=50, value=10, step=1)
    nonstationary = st.sidebar.checkbox("Non-stationary (random walk)", value=False)

    st.sidebar.subheader("Agent strategy")
    strategy = st.sidebar.selectbox(
        "Strategy",
        ["epsilon_greedy", "optimistic_greedy", "ucb"],
        index=0,
        help="epsilon_greedy = ε-greedy, optimistic_greedy = greedy with Q1=+5, ucb = upper confidence bound"
    )

    st.sidebar.subheader("Learning / step-size")
    step_mode = st.sidebar.selectbox(
        "Step-size mode",
        ["sample_average", "constant_alpha"],
        index=0,
        help="sample_average uses α = 1/N(a). constant_alpha uses α = constant."
    )

    epsilon = st.sidebar.slider("ε (epsilon)", 0.0, 0.5, 0.1, 0.01)
    alpha = st.sidebar.slider("α (constant step-size)", 0.01, 1.0, 0.1, 0.01)
    c = st.sidebar.slider("c (UCB exploration)", 0.0, 5.0, 2.0, 0.1)
    initial_q = st.sidebar.slider("Initial Q values (Q1)", -1.0, 10.0, 0.0, 0.5)

    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000_000, value=1234, step=1)

    # smart defaults for optimistic strategy
    if strategy == "optimistic_greedy":
        initial_q = 5.0
        epsilon = 0.0

    return {
        "mode": mode,
        "k": int(k),
        "nonstationary": bool(nonstationary),
        "strategy": strategy,
        "step_mode": step_mode,
        "epsilon": float(epsilon),
        "alpha": float(alpha),
        "c": float(c),
        "initial_q": float(initial_q),
        "seed": int(seed),
    }


def ensure_manual_state(cfg: AgentConfig, seed: int, nonstationary: bool) -> None:
    if "manual_rng" not in st.session_state or st.session_state.get("manual_seed") != seed:
        st.session_state.manual_rng = np.random.default_rng(seed)
        st.session_state.manual_seed = seed
        st.session_state.manual_t = 1

    # bandit
    if "manual_bandit" not in st.session_state or st.session_state.get("manual_reset_bandit", False):
        rng = st.session_state.manual_rng
        if nonstationary:
            st.session_state.manual_bandit = RandomWalkGaussianBandit.new(cfg.k, rng)
        else:
            st.session_state.manual_bandit = GaussianBandit.new(cfg.k, rng)
        st.session_state.manual_reset_bandit = False

    # agent
    if "manual_agent" not in st.session_state or st.session_state.get("manual_reset_agent", False):
        st.session_state.manual_agent = BanditAgent(cfg, st.session_state.manual_rng)
        st.session_state.manual_reset_agent = False


def manual_mode(side: Dict) -> None:
    st.title("Manual vs AI (Interactive)")

    cfg = AgentConfig(
        k=side["k"],
        strategy=side["strategy"],
        step_mode=side["step_mode"],
        epsilon=side["epsilon"],
        alpha=side["alpha"],
        c=side["c"],
        initial_q=side["initial_q"],
    )

    ensure_manual_state(cfg, seed=side["seed"], nonstationary=side["nonstationary"])
    bandit = st.session_state.manual_bandit
    agent = st.session_state.manual_agent
    t = int(st.session_state.manual_t)

    colA, colB, colC = st.columns([1, 1, 1])

    with colA:
        if st.button("🔄 New hidden bandit (new q*)"):
            st.session_state.manual_reset_bandit = True
            st.session_state.manual_t = 1
            st.rerun()

    with colB:
        if st.button("🧠 Reset agent memory (Q and N)"):
            st.session_state.manual_reset_agent = True
            st.session_state.manual_t = 1
            st.rerun()

    with colC:
        reveal = st.toggle("Reveal true q*(a) (for checking)", value=False)

    st.markdown("---")

    # AI suggestion (based on current agent state BEFORE your click)
    suggested_action, suggest_info = agent.choose_action(t)
    st.subheader("AI suggestion (before you click)")
    st.write(f"**AI would pick Arm {suggested_action + 1}**")
    st.info(suggest_info.get("why", ""))
    if suggest_info.get("strategy") == "UCB" and "ucb_score" in suggest_info:
        df_ucb = pd.DataFrame({
            "Arm": np.arange(1, cfg.k + 1),
            "Q(a)": suggest_info["Q_snapshot"],
            "N(a)": suggest_info["N_snapshot"],
            "Bonus": suggest_info["bonus"],
            "UCB Score": suggest_info["ucb_score"],
        })
        st.caption("UCB scores used to choose the action.")
        st.dataframe(df_ucb, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Choose an arm (you are the learner now)")

    # Buttons grid
    grid_cols = st.columns(5 if cfg.k >= 10 else min(cfg.k, 5))
    clicked_action: Optional[int] = None

    for i in range(cfg.k):
        with grid_cols[i % len(grid_cols)]:
            if st.button(f"Arm {i+1}", key=f"arm_btn_{i}"):
                clicked_action = i

    if clicked_action is not None:
        r = bandit.reward(clicked_action)
        update_info = agent.update(clicked_action, r)

        # for non-stationary
        if side["nonstationary"] and isinstance(bandit, RandomWalkGaussianBandit):
            bandit.drift()

        st.session_state.manual_t = t + 1

        st.success(f"✅ You picked Arm {clicked_action + 1}. Reward received: **{r:.4f}**")

        # Show the math
        st.subheader("Update math (incremental learning)")
        st.code(
            f"Old Q = {update_info['old_Q']:.6f}\n"
            f"N(a) after = {update_info['N_after']}\n"
            f"α (step-size) = {update_info['alpha']:.6f}\n"
            f"Error = R - Old Q = {update_info['error']:.6f}\n"
            f"New Q = Old Q + α * Error = {update_info['new_Q']:.6f}\n",
            language="text",
        )

    # Show current agent table
    st.markdown("---")
    st.subheader("Current agent memory (Q and N)")

    table = pd.DataFrame({
        "Arm": np.arange(1, cfg.k + 1),
        "N(a) (count)": agent.N,
        "Q(a) (estimate)": agent.Q,
    })

    if reveal:
        table["q*(a) (true)"] = bandit.q_true
        table["is optimal?"] = table["Arm"].apply(lambda x: "✅" if (x - 1) == bandit.optimal_action() else "")

    st.dataframe(table, use_container_width=True, hide_index=True)

    st.caption(
        "Tip: Try ε=0 (greedy) and see how you/AI can get stuck. Then switch to ε=0.1 or UCB."
    )


def simulation_mode(side: Dict) -> None:
    st.title("Simulation Run (Plots like Figures 2.2–2.4)")

    st.write(
        "This runs many independent bandit problems (runs) and averages performance over time "
        "(Average Reward and % Optimal Action), like the book’s testbed experiments."
    )

    steps = st.number_input("Steps per run", min_value=10, max_value=50_000, value=1_000, step=50)
    runs = st.number_input("Number of runs", min_value=10, max_value=20_000, value=2_000, step=100)

    st.subheader("Choose comparison preset")
    preset = st.selectbox(
        "Preset",
        [
            "Figure 2.2 style: ε-greedy (ε=0, 0.01, 0.1) with sample-average",
            "Figure 2.3 style: Optimistic greedy (Q1=5, ε=0) vs realistic ε=0.1 (α=0.1 constant)",
            "Figure 2.4 style: UCB (c=2) vs ε-greedy (ε=0.1) with sample-average",
            "Custom compare (pick methods below)",
        ],
        index=0
    )

    seed = side["seed"]
    nonstationary = side["nonstationary"]
    k = side["k"]

    methods: List[Tuple[AgentConfig, str]] = []

    if preset.startswith("Figure 2.2"):
        # ε in {0, 0.01, 0.1}, sample-average, Q1=0
        for eps in [0.0, 0.01, 0.1]:
            cfg = AgentConfig(k=k, strategy="epsilon_greedy", step_mode="sample_average", epsilon=eps, initial_q=0.0)
            methods.append((cfg, f"ε-greedy ε={eps:g} (sample avg)"))

    elif preset.startswith("Figure 2.3"):
        # optimistic greedy: Q1=5, ε=0, α=0.1 constant
        cfg1 = AgentConfig(k=k, strategy="optimistic_greedy", step_mode="constant_alpha", alpha=0.1, initial_q=5.0, epsilon=0.0)
        methods.append((cfg1, "Optimistic greedy (Q1=5, ε=0, α=0.1)"))

        # realistic ε-greedy: Q1=0, ε=0.1, α=0.1 constant
        cfg2 = AgentConfig(k=k, strategy="epsilon_greedy", step_mode="constant_alpha", alpha=0.1, epsilon=0.1, initial_q=0.0)
        methods.append((cfg2, "Realistic ε-greedy (Q1=0, ε=0.1, α=0.1)"))

    elif preset.startswith("Figure 2.4"):
        # UCB c=2 vs ε=0.1, both sample-average
        cfg1 = AgentConfig(k=k, strategy="ucb", step_mode="sample_average", c=2.0, initial_q=0.0)
        methods.append((cfg1, "UCB (c=2, sample avg)"))

        cfg2 = AgentConfig(k=k, strategy="epsilon_greedy", step_mode="sample_average", epsilon=0.1, initial_q=0.0)
        methods.append((cfg2, "ε-greedy (ε=0.1, sample avg)"))

    else:
        st.subheader("Custom compare")
        st.write("Add up to 3 methods for comparison (uses your sidebar values as defaults).")

        # Method 1: sidebar method
        cfg_side = AgentConfig(
            k=k,
            strategy=side["strategy"],
            step_mode=side["step_mode"],
            epsilon=side["epsilon"],
            alpha=side["alpha"],
            c=side["c"],
            initial_q=side["initial_q"],
        )
        methods.append((cfg_side, f"Sidebar method ({side['strategy']})"))

        # Method 2: classic ε=0.1 sample avg
        cfg_eps = AgentConfig(k=k, strategy="epsilon_greedy", step_mode="sample_average", epsilon=0.1, initial_q=0.0)
        methods.append((cfg_eps, "ε-greedy (ε=0.1, sample avg)"))

        # Method 3: UCB c=2
        cfg_ucb = AgentConfig(k=k, strategy="ucb", step_mode="sample_average", c=2.0, initial_q=0.0)
        methods.append((cfg_ucb, "UCB (c=2, sample avg)"))

    if st.button("▶ Run simulation"):
        results: List[SimResult] = []
        for cfg, label in methods:
            res = run_many(cfg, steps=int(steps), runs=int(runs), seed=seed, nonstationary=nonstationary, label=label)
            results.append(res)

        # Average reward plot
        fig1 = plt.figure()
        for res in results:
            plt.plot(res.avg_reward, label=res.label)
        plt.title("Average reward over time")
        plt.xlabel("Steps")
        plt.ylabel("Average reward")
        plt.legend()
        plt.grid(True, alpha=0.25)
        st.pyplot(fig1)

        # % optimal action plot
        fig2 = plt.figure()
        for res in results:
            plt.plot(res.pct_optimal, label=res.label)
        plt.title("% Optimal action over time")
        plt.xlabel("Steps")
        plt.ylabel("% Optimal action")
        plt.legend()
        plt.grid(True, alpha=0.25)
        st.pyplot(fig2)

        st.caption(
            "Note: If you enable non-stationary random walk, sample-average methods often degrade, "
            "and constant-α usually adapts better."
        )


def main():
    side = ui_sidebar()

    if side["mode"] == "Manual vs AI (Interactive)":
        manual_mode(side)
    else:
        simulation_mode(side)

    st.markdown("---")
    st.markdown(
        "Educational mapping to the PDF: "
        "- Q estimates (sample average) Eq (2.1), greedy Eq (2.2), incremental update Eq (2.3)/(2.4), "
        "constant α Eq (2.5), optimistic Q1=+5, and UCB Eq (2.10)."
    )


if __name__ == "__main__":
    main()
