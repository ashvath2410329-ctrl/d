import numpy as np
import matplotlib.pyplot as plt

maze = [
    "S.#.....",
    ".##.###.",
    "....#...",
    ".####.#.",
    "......#.",
    ".####.#G",
    "....#...",
    "###...#.",
]
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
arrows = ["↑", "↓", "←", "→"]
start, goal = (0, 0), (5, 7)
walls = np.array([[ch == "#" for ch in row] for row in maze])

alpha, gamma = 0.1, 0.95
eps_start, eps_end, eps_decay = 1.0, 0.01, 0.995
episodes, max_steps = 2000, 200
rng = np.random.default_rng(42)


def step(state, action):
    r = state[0] + moves[action][0]
    c = state[1] + moves[action][1]
    if r < 0 or r > 7 or c < 0 or c > 7 or maze[r][c] == "#":
        return state, -5.0, False
    if (r, c) == goal:
        return (r, c), 100.0, True
    return (r, c), -1.0, False


def choose_action(state, epsilon):
    if rng.random() < epsilon:
        return int(rng.integers(4))
    q = Q[state]
    return int(rng.choice(np.flatnonzero(q == q.max())))


Q = np.zeros((8, 8, 4))
epsilon = eps_start
rewards, steps_per_ep = [], []

for ep in range(episodes):
    state = start
    total, steps, done = 0.0, 0, False

    while not done and steps < max_steps:
        action = choose_action(state, epsilon)
        next_state, reward, done = step(state, action)
        best_next = 0.0 if done else np.max(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])
        state = next_state
        total += reward
        steps += 1

    epsilon = max(eps_end, epsilon * eps_decay)
    rewards.append(total)
    steps_per_ep.append(steps)

    if (ep + 1) % 200 == 0:
        print(f"Episode {ep + 1:5d} | avg reward (last 200): {np.mean(rewards[-200:]):7.1f} | "
              f"avg steps: {np.mean(steps_per_ep[-200:]):6.1f} | epsilon: {epsilon:.3f}")

state, path = start, [start]
for _ in range(64):
    action = int(np.argmax(Q[state]))
    state, _, done = step(state, action)
    path.append(state)
    if done:
        break

print("\nGreedy path reached goal:", state == goal)
print("Path length (steps):", len(path) - 1)
print("Path:", path)


def moving_avg(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode="valid")


fig, axes = plt.subplots(1, 3, figsize=(17, 5))

axes[0].plot(rewards, alpha=0.3, label="Episode reward")
axes[0].plot(range(49, len(rewards)), moving_avg(rewards), label="Moving avg (50)")
axes[0].set(title="Reward per episode", xlabel="Episode", ylabel="Total reward")
axes[0].legend()

axes[1].plot(steps_per_ep, alpha=0.3, label="Steps")
axes[1].plot(range(49, len(steps_per_ep)), moving_avg(steps_per_ep), label="Moving avg (50)")
axes[1].set(title="Steps to finish episode", xlabel="Episode", ylabel="Steps")
axes[1].legend()

ax = axes[2]
ax.imshow(walls.astype(float), cmap="Greys", vmin=0, vmax=1.5)
for r in range(8):
    for c in range(8):
        if walls[r, c]:
            continue
        if (r, c) == start:
            ax.text(c, r, "S", ha="center", va="center", color="green", fontsize=14, fontweight="bold")
        elif (r, c) == goal:
            ax.text(c, r, "G", ha="center", va="center", color="red", fontsize=14, fontweight="bold")
        elif Q[r, c].max() != 0:
            ax.text(c, r, arrows[int(np.argmax(Q[r, c]))], ha="center", va="center",
                    color="steelblue", fontsize=14)

ys, xs = zip(*path)
ax.plot(xs, ys, "o-", color="orange", linewidth=2, markersize=4, alpha=0.8, label="Greedy path")
ax.set_xticks(range(8))
ax.set_yticks(range(8))
ax.set_title("Learned policy & optimal path")
ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("qlearning_maze_results.png", dpi=150)
plt.show()
