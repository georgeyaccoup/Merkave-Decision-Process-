"""
robot_dp_tutor.py
A step-by-step, terminal-guided tutorial:
Dynamic Programming (DP) on a robotics MDP (grid navigation with slip).

What you'll learn:
1) How to define an MDP model (S, A, P, R, gamma)
2) How to build the transition table P[s][a] (the "model")
3) How Value Iteration works (Bellman optimality backups)
4) How to extract a greedy policy (navigation arrows)

This script is intentionally verbose + interactive.
"""

from math import inf

# =========================
# Small helper: step pauses
# =========================
def pause(step_title: str):
    print("\n" + "=" * 70)
    print(step_title)
    print("=" * 70)
    input("Press ENTER to continue...")

# =========================
# Pretty printing utilities
# =========================
def print_grid_text(grid):
    print("Grid legend: S=start (just for display), G=goal (terminal), #=obstacle, .=free\n")
    for row in grid:
        print(" ".join(row))
    print()

def print_values_as_grid(grid, state_id, V):
    """Print V(s) aligned with the grid map."""
    ROWS, COLS = len(grid), len(grid[0])
    for r in range(ROWS):
        row_str = []
        for c in range(COLS):
            cell = grid[r][c]
            if cell == '#':
                row_str.append("#####")
            elif cell == 'G':
                row_str.append("  G  ")
            else:
                s = state_id[(r, c)]
                row_str.append(f"{V[s]:5.1f}")
        print(" ".join(row_str))
    print()

def print_policy_as_grid(grid, state_id, pi, A_NAME):
    """Print policy arrows aligned with the grid map."""
    ROWS, COLS = len(grid), len(grid[0])
    for r in range(ROWS):
        row_str = []
        for c in range(COLS):
            cell = grid[r][c]
            if cell == '#':
                row_str.append("#")
            elif cell == 'G':
                row_str.append("G")
            else:
                s = state_id[(r, c)]
                row_str.append(A_NAME[pi[s]])
        print(" ".join(row_str))
    print()

# ============================================================
# STEP 1) Define the robotics problem as a gridworld navigation
# ============================================================
pause("STEP 1 — Define the robotics problem (warehouse navigation)")

grid = [
    ['S', '.', '.', '#', '.'],
    ['.', '#', '.', '#', '.'],
    ['.', '#', '.', '.', '.'],
    ['.', '.', '.', '#', '.'],
    ['#', '.', '.', '.', 'G'],
]

print("We model a mobile robot navigating a warehouse to a charging station.")
print("The world is a 5x5 grid with obstacles (#) and a terminal goal (G).")
print_grid_text(grid)

# ======================================================
# STEP 2) Define MDP components: states, actions, rewards
# ======================================================
pause("STEP 2 — Define the MDP components (S, A, R, gamma)")

ROWS, COLS = len(grid), len(grid[0])

# Actions: 4 high-level motion commands
U, R, D, L = 0, 1, 2, 3
ACTIONS = [U, R, D, L]
A_NAME = {U: "U", R: "R", D: "D", L: "L"}

print("Actions (A): 4 commands -> Up (U), Right (R), Down (D), Left (L)")
print("Why 4 actions? Because in robotics we often discretize motion for planning.\n")

# Rewards: time/energy cost per step (common in robotics)
STEP_REWARD = -1.0    # each move costs time/energy
GOAL_REWARD = 0.0     # reaching goal gives 0 and terminates
GAMMA = 0.9           # discount factor

print(f"Reward design (R):")
print(f"  - Every step reward = {STEP_REWARD}  (penalize time/energy)")
print(f"  - Goal reward       = {GOAL_REWARD}  (terminal)\n")
print(f"Discount gamma = {GAMMA}")
print("Gamma < 1 helps convergence and prefers shorter paths.\n")

# ==========================================================
# STEP 3) Add robotics realism: uncertainty (slip / actuation)
# ==========================================================
pause("STEP 3 — Add motion uncertainty (transition probabilities)")

# Slip model: when you command an action, robot might slip to adjacent directions.
P_INTENDED = 0.8
P_LEFT = 0.1
P_RIGHT = 0.1

print("Robotics realism: motion is noisy (wheel slip, uneven floor, etc.)")
print("Transition model p(s'|s,a):")
print(f"  - {P_INTENDED:.1f} go intended direction")
print(f"  - {P_LEFT:.1f} slip to LEFT of intended direction")
print(f"  - {P_RIGHT:.1f} slip to RIGHT of intended direction\n")

def in_bounds(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

def is_wall(r, c):
    return grid[r][c] == '#'

def is_goal(r, c):
    return grid[r][c] == 'G'

def move(r, c, a):
    """
    Attempt to move from (r,c) using action a.
    If next cell is outside map or obstacle -> stay in place (robot blocked).
    """
    if a == U:
        nr, nc = r - 1, c
    elif a == D:
        nr, nc = r + 1, c
    elif a == L:
        nr, nc = r, c - 1
    else:  # R
        nr, nc = r, c + 1

    # Collision / boundary => no motion
    if not in_bounds(nr, nc) or is_wall(nr, nc):
        return r, c
    return nr, nc

def left_of(a):
    # relative left turn in action space
    return (a - 1) % 4

def right_of(a):
    # relative right turn in action space
    return (a + 1) % 4

# ===========================================================
# STEP 4) Build state indexing: map (r,c) <-> state integer s
# ===========================================================
pause("STEP 4 — Build the state space (indexing states)")

state_id = {}   # (r,c) -> s
id_state = {}   # s -> (r,c)

s = 0
for r in range(ROWS):
    for c in range(COLS):
        if is_wall(r, c):
            continue
        state_id[(r, c)] = s
        id_state[s] = (r, c)
        s += 1

N_S = s
N_A = 4

def is_terminal_state(s):
    r, c = id_state[s]
    return is_goal(r, c)

print(f"We created {N_S} states (we exclude obstacle cells).")
print("Now every free cell (including S and G) has a state index s.\n")
print("Example: some state ids:")
examples = list(state_id.items())[:6]
for (r, c), sid in examples:
    print(f"  cell ({r},{c}) -> state {sid}")
print()

# ===========================================================
# STEP 5) Build the MDP MODEL P[s][a] -> list of transitions
# ===========================================================
pause("STEP 5 — Build the transition model P[s][a] (this is the MDP 'model')")

# P[s][a] = list of (prob, s_next, reward, done)
P = [[[] for _ in range(N_A)] for _ in range(N_S)]

for s in range(N_S):
    r, c = id_state[s]
    for a in ACTIONS:

        # Terminal: stay in terminal forever (common episodic modeling)
        if is_terminal_state(s):
            P[s][a] = [(1.0, s, 0.0, True)]
            continue

        # stochastic outcomes: intended, slip-left, slip-right
        outcomes = [
            (P_INTENDED, a),
            (P_LEFT, left_of(a)),
            (P_RIGHT, right_of(a)),
        ]

        # Merge outcomes that end up in same next state
        agg = {}  # s2 -> [prob_sum, reward, done]
        for prob, a2 in outcomes:
            nr, nc = move(r, c, a2)
            s2 = state_id[(nr, nc)]
            done = is_goal(nr, nc)
            reward = GOAL_REWARD if done else STEP_REWARD

            if s2 not in agg:
                agg[s2] = [0.0, reward, done]
            agg[s2][0] += prob

        P[s][a] = [(p, s2, rew, done) for s2, (p, rew, done) in agg.items()]

print("✅ Transition model built.")
print("This is the key DP requirement: we know p(s',r|s,a).\n")

# Show an example transition for learning
sample_cell = (0, 0)  # likely 'S' in our grid
sample_state = state_id[sample_cell]
sample_action = R      # try moving Right from start

print(f"Example transitions from cell {sample_cell} (state {sample_state}) when action=RIGHT:")
for prob, s2, rew, done in P[sample_state][sample_action]:
    print(f"  prob={prob:.2f} -> next_state={s2} (cell {id_state[s2]}), reward={rew}, done={done}")
print()

# ===========================================================
# STEP 6) Value Iteration: DP to compute the optimal value V*
# ===========================================================
pause("STEP 6 — Value Iteration (DP): compute V*(s) using Bellman optimality backups")

print("Value Iteration idea:")
print("  We maintain a value V(s). Initially V(s)=0.")
print("  Then we repeatedly apply the Bellman optimality update:")
print("     V(s) <- max_a sum_{s'} P(s'|s,a) [ R + gamma * V(s') ]")
print("  We repeat until values stop changing (convergence).\n")

def value_iteration(theta=1e-6, max_iters=10_000, show_first_k_sweeps=3):
    """
    theta: convergence threshold
    show_first_k_sweeps: educational printing for early iterations
    """
    V = [0.0] * N_S

    for it in range(1, max_iters + 1):
        delta = 0.0

        # sweep over all states
        for s in range(N_S):
            if is_terminal_state(s):
                continue

            v_old = V[s]
            best = -inf

            # compute best action value
            for a in ACTIONS:
                q = 0.0
                for prob, s2, r, done in P[s][a]:
                    q += prob * (r + GAMMA * V[s2])
                best = max(best, q)

            V[s] = best
            delta = max(delta, abs(v_old - V[s]))

        # Teaching: show the first few sweeps so you SEE the process
        if it <= show_first_k_sweeps:
            print(f"\n--- After sweep #{it} ---  (max change delta = {delta:.6f})")
            print_values_as_grid(grid, state_id, V)

        # stop condition
        if delta < theta:
            print(f"Converged at sweep #{it} (delta={delta:.6f} < theta={theta}).\n")
            return V, it

    print(f"Stopped at max_iters={max_iters} (may or may not be fully converged).")
    return V, max_iters

V_star, sweeps = value_iteration(theta=1e-6, show_first_k_sweeps=3)

print("Final converged V*(s):")
print_values_as_grid(grid, state_id, V_star)

# ===========================================================
# STEP 7) Extract a policy from V*: greedy policy (navigation)
# ===========================================================
pause("STEP 7 — Extract the optimal policy pi*(s) from V*(s)")

print("Now that we have V*(s), we can get a policy:")
print("  pi(s) = argmax_a sum_{s'} P(s'|s,a) [ R + gamma * V*(s') ]")
print("This creates a navigation arrow (U/R/D/L) for each cell.\n")

def greedy_policy_from_V(V):
    pi = [R] * N_S
    for s in range(N_S):
        if is_terminal_state(s):
            continue

        best_a = None
        best_q = -inf
        for a in ACTIONS:
            q = 0.0
            for prob, s2, r, done in P[s][a]:
                q += prob * (r + GAMMA * V[s2])
            if q > best_q:
                best_q = q
                best_a = a

        pi[s] = best_a
    return pi

pi_star = greedy_policy_from_V(V_star)

print("Optimal policy arrows (pi*):")
print_policy_as_grid(grid, state_id, pi_star, A_NAME)

# ===========================================================
# STEP 8) Learner exercises: change parameters and observe
# ===========================================================
pause("STEP 8 — Exercises (so you can do it alone next time)")

print("✅ You now have a complete DP solution pipeline.")
print("\nTry these changes yourself (edit the constants near the top):")
print("1) Make the robot more slippery:")
print("   - Try P_INTENDED=0.6, P_LEFT=0.2, P_RIGHT=0.2")
print("   - Observe how the policy becomes more 'cautious' near obstacles.\n")
print("2) Change reward shaping:")
print("   - Try GOAL_REWARD=+20 instead of 0")
print("   - Try adding obstacle-collision penalty (more advanced)\n")
print("3) Change gamma:")
print("   - gamma=0.99 makes it care more about long-term")
print("   - gamma=0.5 makes it super short-term\n")
print("4) Change the map:")
print("   - Add/remove # obstacles and see the policy update.\n")
print("When you're ready, next we can implement Policy Iteration (evaluate + improve).")
# ===========================================================
# STEP 9) Policy Iteration: Evaluate + Improve until stable
# ===========================================================
pause("STEP 9 — Policy Iteration (DP): policy evaluation + policy improvement")

print("Policy Iteration idea (classic DP):")
print("  We do two loops repeatedly:")
print("   (A) Policy Evaluation: compute V_pi for the CURRENT policy pi")
print("   (B) Policy Improvement: make pi greedy w.r.t. V_pi")
print("  If policy stops changing => optimal policy found.\n")

# -----------------------------
# Helper: initialize a policy
# -----------------------------
def init_policy(method="all_right"):
    """
    Create an initial policy pi[s].
    method:
      - 'all_right': all states choose RIGHT (simple deterministic start)
      - 'all_down' : all states choose DOWN
      - 'random'   : random action in each state (needs import random)
    """
    pi = [R] * N_S

    if method == "all_down":
        pi = [D] * N_S
    elif method == "random":
        import random
        pi = [random.choice(ACTIONS) for _ in range(N_S)]

    # terminal state's action doesn't matter, but set for cleanliness
    for s in range(N_S):
        if is_terminal_state(s):
            pi[s] = R
    return pi

# ------------------------------------------
# Policy Evaluation (Iterative, step-by-step)
# ------------------------------------------
def policy_evaluation(pi, theta=1e-6, max_sweeps=10_000, show_first_k_sweeps=2):
    """
    Evaluate a fixed policy pi -> returns V_pi
    Using iterative policy evaluation (Bellman expectation backups).

    Update:
      V(s) <- sum_{s'} P(s'|s, pi(s)) [ r + gamma * V(s') ]
    """
    V = [0.0] * N_S

    for sweep in range(1, max_sweeps + 1):
        delta = 0.0

        for s in range(N_S):
            if is_terminal_state(s):
                continue

            v_old = V[s]
            a = pi[s]

            # Expected value under action a = pi(s)
            v_new = 0.0
            for prob, s2, r, done in P[s][a]:
                v_new += prob * (r + GAMMA * V[s2])

            V[s] = v_new
            delta = max(delta, abs(v_old - v_new))

        # Teaching output: show early sweeps to visualize convergence
        if sweep <= show_first_k_sweeps:
            print(f"\n[Policy Evaluation] sweep #{sweep}  (delta={delta:.6f})")
            print_values_as_grid(grid, state_id, V)

        if delta < theta:
            print(f"[Policy Evaluation] converged in {sweep} sweeps (delta={delta:.6f} < theta={theta}).")
            return V, sweep

    print(f"[Policy Evaluation] stopped at max_sweeps={max_sweeps}.")
    return V, max_sweeps

# ------------------------------------------
# Policy Improvement (make policy greedy)
# ------------------------------------------
def policy_improvement(V, pi):
    """
    Improve policy pi by making it greedy w.r.t. current value V.
    Returns (new_pi, stable_flag)
    """
    policy_stable = True

    for s in range(N_S):
        if is_terminal_state(s):
            continue

        old_a = pi[s]

        # compute best action using one-step lookahead with V
        best_a = None
        best_q = -inf
        for a in ACTIONS:
            q = 0.0
            for prob, s2, r, done in P[s][a]:
                q += prob * (r + GAMMA * V[s2])
            if q > best_q:
                best_q = q
                best_a = a

        pi[s] = best_a
        if best_a != old_a:
            policy_stable = False

    return pi, policy_stable

# ------------------------------------------
# Full Policy Iteration Loop (Evaluate + Improve)
# ------------------------------------------
def policy_iteration(theta=1e-6, eval_show_first_k_sweeps=2, max_outer_iters=100):
    """
    Outer loop:
      1) Evaluate current policy -> V_pi
      2) Improve policy -> pi'
      3) Stop if stable
    """
    pi = init_policy(method="all_right")  # you can switch to 'random'
    print("Initial policy (starting guess):")
    print_policy_as_grid(grid, state_id, pi, A_NAME)

    for it in range(1, max_outer_iters + 1):
        print(f"\n--- POLICY ITERATION round #{it} ---")

        # A) Evaluate
        V, sweeps = policy_evaluation(pi, theta=theta, show_first_k_sweeps=eval_show_first_k_sweeps)

        # B) Improve
        pi, stable = policy_improvement(V, pi)

        print("\nImproved policy after this round:")
        print_policy_as_grid(grid, state_id, pi, A_NAME)

        if stable:
            print("✅ Policy is stable (no changes). This policy is optimal.")
            return V, pi, it

    print("⚠️ Reached max_outer_iters without stability (unusual for small problems).")
    return V, pi, max_outer_iters

# Run Policy Iteration
V_pi_star, pi_pi_star, outer_iters = policy_iteration(theta=1e-6, eval_show_first_k_sweeps=2)

print("\nFinal V(s) from Policy Iteration:")
print_values_as_grid(grid, state_id, V_pi_star)

print("Final policy pi(s) from Policy Iteration:")
print_policy_as_grid(grid, state_id, pi_pi_star, A_NAME)

# -----------------------------------------------------------
# STEP 10) Compare with Value Iteration result (sanity check)
# -----------------------------------------------------------
pause("STEP 10 — Compare Policy Iteration vs Value Iteration (sanity check)")

print("We already computed V_star and pi_star using Value Iteration earlier.")
print("Now we compare the greedy policies. They should match (or be equivalent).\n")

same = True
for s in range(N_S):
    if is_terminal_state(s):
        continue
    if pi_pi_star[s] != pi_star[s]:
        same = False
        break

print("Policy Iteration policy:")
print_policy_as_grid(grid, state_id, pi_pi_star, A_NAME)

print("Value Iteration policy:")
print_policy_as_grid(grid, state_id, pi_star, A_NAME)

if same:
    print("✅ Policies match exactly. Great!")
else:
    print("ℹ️ Policies differ in some states. This can happen if multiple actions are equally optimal.")
    print("   Both can still be optimal (ties).")