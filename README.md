# Railroad Environment

A Gymnasium environment for the CodinGame **Summer Challenge 2026** railway game, plus a ladder
of scripted opponents to train and benchmark against.

The rules here are a port of the official Java referee, not an approximation. Track costs,
region inking, the north-first BFS tie-break, the one-disruption-per-turn limit and the map
generator's constants all match the original. An agent that beats a boss here is playing the
same game the real ladder plays.

```bash
pip install -r requirements.txt
jupyter notebook play_vs_bosses.ipynb
```

## The game in one paragraph

Two players share a map of 14-20 rows (width follows a 1.5 aspect ratio, so up to 30 columns).
Towns want to be connected to specific other towns. Each turn you receive **3 paint points** and
spend them placing track, which costs 1 on plains, 2 on river and 3 on mountain. A connection
becomes *active* when a path of track links two towns that want each other, and from then on it
pays **1 point per turn for every one of your tracks lying on that path**. You also get **1
disruption point** per turn: spend it to add instability to a region, and at 4 instability the
region is **inked**, permanently destroying every track in it, yours included. Regions containing
a town can never be inked. Games run 100 turns.

The scoring path is the one with the **fewest cells**, found by breadth-first search. Terrain
cost only affects what you pay to build, never which route the trains take.

## Quick start

```python
from railroad_env import RailroadGymEnv

env = RailroadGymEnv(max_turns=100, opponent_strategy="level2ProMax", seed=0)
obs, info = env.reset()                     # obs: (height, width, 28) float32

while not env.game_state.is_done():
    obs, reward, terminated, truncated, info = env.step({
        "actions": [("PLACE", 4, 7), ("DISRUPT", 12)]
    })

print(env.game_state.scores)                # [you, opponent]
```

You are always player 0 and the scripted opponent is player 1.

### Actions

| action | meaning |
|---|---|
| `("PLACE", x, y)` | place one track, costs 1-3 paint by terrain |
| `("AUTOPLACE", x1, y1, x2, y2)` | build the cheapest chain between two points, one per turn |
| `("DISRUPT", region_id)` | add 1 instability to a region, one per turn |
| `("WAIT",)` | do nothing |

Emit as many `PLACE` actions as your paint affords. Illegal actions are ignored rather than
raising, exactly as the referee reports them as errors and continues.

Useful guards before acting: `game_state._is_placeable(x, y)`, `game_state.can_disrupt(zone_id)`
and `game_state.get_track_cost(x, y)`.

## The opponents

`BOSS_TIERS` lists them weakest to strongest. The first three are ports of the challenge's own
shipped AIs; the `Pro` tiers are ours and are harder than anything the real ladder fields.

| name | behaviour |
|---|---|
| `level1` | always waits. The shipped League 1 boss |
| `level1Pro` | places randomly half the time, disrupts half the time |
| `level2` | AUTOPLACE between **two random towns**, every turn. The shipped League 2 boss |
| `level2Pro` | prices every unfinished connection and commits to the cheapest. Never disrupts |
| `level2ProMax` | `level2Pro`, plus it inks the region where your track most outnumbers its own |

`level2ProMax` is a large jump. It finishes its own network in roughly 25 turns and then spends
the remaining 75 doing nothing but destroying your regions, one every four turns, while its
completed connections keep paying. Beating it needs a different strategy, not just better
building: regions holding a town are the only permanently safe ground, and they are about 14% of
the board.

## The observation

`(height, width, 28)` float32, from player 0's point of view. Channel 3 is always the enemy and
channel 4 is always you, whichever index you hold.

| channel | contents |
|---|---|
| 0-2 | terrain one-hot: plains, river, mountain |
| 3, 4 | enemy track present, own track present |
| 5, 6 | enemy / own tracks in this cell's region, **count / 10** |
| 7 | region size / board cells |
| 8, 9 | enemy / own tracks in region **on an active connection**, count / 10 |
| 10-21 | per-town route guides: `-1` this town, `-0.5` a town it wants, `+1` cells between |
| 22 | region instability / 4, so 1.0 means inked next disruption |
| 23 | region inked |
| 24 | cell is part of an active connection |
| 25 | a town stands here (ownerless, never buildable) |
| 26 | (own score - enemy score) / 10000, the same value in every cell |
| 27 | turn / 100, the same value in every cell |

Two deliberate choices worth knowing. Channels 5, 6, 8 and 9 are **counts, not densities**,
because the quantities the game acts on are absolute: a disruptor compares track counts, and an
active connection pays per track. And channels 26 and 27 are flat planes because a convolution is
local and has no other way to see a board-wide scalar.

Route guides use unweighted BFS on open terrain, blocked only by inked regions. They ignore
existing track, so they answer "where would a route go on an empty board", not "where is the
paying route right now".

### Feeding it to a network

Maps vary in size, so `encoding.py` centres the board on a fixed 20x30 canvas and marks the
padding as inked, giving a constant `(28, 20, 30)` tensor:

```python
from encoding import encode_state, build_place_mask, build_disrupt_mask

state = encode_state(env.game_state)          # (28, 20, 30) channels-first
mask  = build_place_mask(env.game_state, paint_budget=3)
```

Use the masks. Most cells are illegal on any given turn, and an unmasked policy spends its
capacity learning that instead of learning to play.

## Notes

`pygame` is only needed to watch a game and is imported on first use, so training works without
it. `render_mode="human"` will ask for it.

Maps are procedurally generated per seed. Fixing the seed set is what makes two evaluations
comparable; changing it means you are measuring the maps, not the agent.
