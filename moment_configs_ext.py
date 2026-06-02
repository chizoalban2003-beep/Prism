"""
moment_configs_ext.py
=====================
KDE Moment Platform — Extended Sport Moment Configurations

Adds 11 new MomentSportConfig entries to ALL_MOMENT_CONFIGS covering:
  Tennis      — serve_deuce, serve_ad, baseline_rally, net_approach
  Basketball  — pick_roll_read, post_up, fast_break
  Rugby Union — ball_carrier_contact, breakdown_contest, lineout_call
  Cricket     — batting_delivery

Call ``register_extended_configs()`` once at application startup to merge
these configs into the global registry.  Do NOT modify moment_analyzer.py.
"""

from __future__ import annotations

from moment_analyzer import ALL_MOMENT_CONFIGS, MomentOption, MomentSportConfig

# ── Tennis ────────────────────────────────────────────────────────────────────

TENNIS_SERVE_DEUCE = MomentSportConfig(
    sport="Tennis", moment_type="serve_deuce", bandwidth=0.19,
    options=[
        MomentOption("Wide to deuce",       0.00, 70,  4, 18, 0.68),
        MomentOption("Body serve",          0.17, 60,  4, 14, 0.76),
        MomentOption("Kick to backhand",    0.33, 65,  4, 20, 0.72),
        MomentOption("Flat down T",         0.52, 85,  5, 28, 0.58),
        MomentOption("Wide flat ace bid",   0.70, 95,  5, 38, 0.46),
        MomentOption("Slice out wide",      0.85, 75,  4, 24, 0.62),
        MomentOption("Second-serve Panenka",1.00, 50,  4, 45, 0.82),
    ],
)

TENNIS_SERVE_AD = MomentSportConfig(
    sport="Tennis", moment_type="serve_ad", bandwidth=0.19,
    options=[
        MomentOption("Wide to ad court",    0.00, 80,  4, 20, 0.65),
        MomentOption("Body serve",          0.18, 60,  4, 14, 0.76),
        MomentOption("Kick to forehand",    0.36, 70,  4, 22, 0.68),
        MomentOption("Flat down T",         0.54, 90,  5, 30, 0.55),
        MomentOption("Wide ace bid",        0.72, 100, 5, 40, 0.44),
        MomentOption("Slice T",             0.88, 80,  4, 26, 0.60),
        MomentOption("Second kick",         1.00, 50,  4, 42, 0.84),
    ],
)

TENNIS_BASELINE_RALLY = MomentSportConfig(
    sport="Tennis", moment_type="baseline_rally", bandwidth=0.18,
    options=[
        MomentOption("High defensive lob",  0.00, 20,  3, 10, 0.94),
        MomentOption("Deep cross reset",    0.15, 50,  4, 14, 0.88),
        MomentOption("Slice neutralise",    0.28, 55,  4, 16, 0.86),
        MomentOption("Topspin cross",       0.43, 80,  5, 22, 0.78),
        MomentOption("Inside-out forehand", 0.57, 105, 6, 32, 0.64),
        MomentOption("Down the line",       0.70, 125, 7, 40, 0.52),
        MomentOption("Drop shot",           0.83, 140, 7, 52, 0.44),
        MomentOption("Outright winner",     1.00, 160, 8, 62, 0.36),
    ],
)

TENNIS_NET_APPROACH = MomentSportConfig(
    sport="Tennis", moment_type="net_approach", bandwidth=0.18,
    options=[
        MomentOption("Drop back",           0.00, 15,  3,  8, 0.94),
        MomentOption("Defensive volley",    0.18, 50,  4, 16, 0.82),
        MomentOption("Deep approach shot",  0.35, 75,  5, 22, 0.72),
        MomentOption("Slice approach",      0.50, 85,  5, 26, 0.68),
        MomentOption("Aggressive approach", 0.65, 110, 6, 36, 0.56),
        MomentOption("Swinging volley",     0.80, 140, 7, 48, 0.44),
        MomentOption("Winner passing shot", 1.00, 170, 8, 60, 0.32),
    ],
)

# ── Basketball ────────────────────────────────────────────────────────────────

BASKETBALL_PICK_ROLL = MomentSportConfig(
    sport="Basketball", moment_type="pick_roll_read", bandwidth=0.17,
    options=[
        MomentOption("Reset / re-screen",      0.00, 20,  3,  8, 0.94),
        MomentOption("Pass back to PG",        0.14, 55,  4, 14, 0.90),
        MomentOption("Kick to corner 3",       0.28, 95,  4, 22, 0.78),
        MomentOption("Hit roll man alley-oop", 0.44, 105, 5, 28, 0.70),
        MomentOption("Mid-range pull-up",      0.58, 75,  5, 30, 0.50),
        MomentOption("Drive to rim",           0.72, 100, 6, 36, 0.58),
        MomentOption("Pocket pass paint",      0.85, 110, 5, 32, 0.64),
        MomentOption("Step-back 3",            1.00, 120, 7, 48, 0.36),
    ],
)

BASKETBALL_POST_UP = MomentSportConfig(
    sport="Basketball", moment_type="post_up", bandwidth=0.18,
    options=[
        MomentOption("Pass out reset",         0.00, 20,  3,  8, 0.94),
        MomentOption("Kick to 3pt shooter",    0.14, 95,  4, 22, 0.76),
        MomentOption("Drop-step baseline",     0.30, 80,  5, 26, 0.62),
        MomentOption("Face-up mid-range",      0.46, 70,  5, 28, 0.55),
        MomentOption("Up-and-under",           0.60, 90,  6, 32, 0.58),
        MomentOption("Jump hook",              0.74, 100, 6, 36, 0.52),
        MomentOption("Power move",             0.88, 110, 7, 42, 0.48),
        MomentOption("And-one attempt",        1.00, 130, 7, 48, 0.44),
    ],
)

BASKETBALL_FAST_BREAK = MomentSportConfig(
    sport="Basketball", moment_type="fast_break", bandwidth=0.15,
    options=[
        MomentOption("Slow down / set",       0.00, 20,  3,  8, 0.96),
        MomentOption("Pull up short",         0.14, 55,  4, 18, 0.55),
        MomentOption("Kick to trailer",       0.28, 95,  4, 22, 0.78),
        MomentOption("Lay-up right",          0.44, 90,  4, 20, 0.75),
        MomentOption("Lay-up left",           0.58, 90,  4, 20, 0.72),
        MomentOption("Dunk attempt",          0.74, 120, 5, 28, 0.65),
        MomentOption("Alley-oop lob",         0.88, 130, 5, 32, 0.62),
        MomentOption("And-one drive",         1.00, 145, 6, 40, 0.50),
    ],
)

# ── Rugby Union ───────────────────────────────────────────────────────────────

RUGBY_BALL_CARRIER_HIT = MomentSportConfig(
    sport="Rugby Union", moment_type="ball_carrier_contact", bandwidth=0.17,
    options=[
        MomentOption("Secure ruck",         0.00,  30, 4, 12, 0.92),
        MomentOption("Pop to support",      0.14,  65, 5, 18, 0.84),
        MomentOption("Drive forward",       0.28,  80, 6, 24, 0.76),
        MomentOption("Offload in tackle",   0.44, 110, 7, 38, 0.54),
        MomentOption("Flat pass wide",      0.58, 120, 7, 32, 0.66),
        MomentOption("Long miss pass",      0.72, 140, 8, 45, 0.52),
        MomentOption("Chip through",        0.85, 160, 9, 58, 0.36),
        MomentOption("Kick for touch",      1.00,  50, 5, 20, 0.82),
    ],
)

RUGBY_BREAKDOWN = MomentSportConfig(
    sport="Rugby Union", moment_type="breakdown_contest", bandwidth=0.18,
    options=[
        MomentOption("Clear out attacker",  0.00, 25,  4, 10, 0.88),
        MomentOption("Jackal attempt",      0.20, 80,  6, 30, 0.48),
        MomentOption("Pilfer from side",    0.40, 90,  6, 35, 0.44),
        MomentOption("Slow the ball",       0.60, 70,  5, 25, 0.58),
        MomentOption("Counter-ruck",        0.78, 85,  6, 32, 0.52),
        MomentOption("Turnover steal",      1.00, 150, 8, 55, 0.28),
    ],
)

RUGBY_LINEOUT = MomentSportConfig(
    sport="Rugby Union", moment_type="lineout_call", bandwidth=0.20,
    options=[
        MomentOption("Safe short call",     0.00,  35, 3, 10, 0.94),
        MomentOption("Middle jumper",       0.20,  70, 5, 18, 0.80),
        MomentOption("Peel move",           0.40,  90, 5, 22, 0.72),
        MomentOption("Back jumper",         0.60, 100, 5, 26, 0.68),
        MomentOption("Driving maul",        0.78, 120, 6, 32, 0.60),
        MomentOption("Dummy and peel",      1.00, 150, 7, 45, 0.42),
    ],
)

# ── Cricket ───────────────────────────────────────────────────────────────────

CRICKET_BATTING_DELIVERY = MomentSportConfig(
    sport="Cricket", moment_type="batting_delivery", bandwidth=0.20,
    options=[
        MomentOption("Defensive block",     0.00,  10,  2,  5, 0.97),
        MomentOption("Leave outside off",   0.12,  15,  2,  8, 0.96),
        MomentOption("Nurdle to leg",       0.25,  30,  3, 10, 0.90),
        MomentOption("Drive on side",       0.40,  60,  5, 18, 0.80),
        MomentOption("Cut shot",            0.54,  90,  6, 28, 0.70),
        MomentOption("Pull shot",           0.66, 110,  7, 38, 0.60),
        MomentOption("Slog sweep",          0.78, 140,  8, 52, 0.48),
        MomentOption("Lofted drive",        0.88, 155,  9, 58, 0.42),
        MomentOption("Helicopter slog",     1.00, 180, 10, 70, 0.35),
    ],
)

# ── Registry ──────────────────────────────────────────────────────────────────

EXTENDED_CONFIGS: dict[tuple[str, str], MomentSportConfig] = {
    ("Tennis",      "serve_deuce"):          TENNIS_SERVE_DEUCE,
    ("Tennis",      "serve_ad"):             TENNIS_SERVE_AD,
    ("Tennis",      "baseline_rally"):       TENNIS_BASELINE_RALLY,
    ("Tennis",      "net_approach"):         TENNIS_NET_APPROACH,
    ("Basketball",  "pick_roll_read"):       BASKETBALL_PICK_ROLL,
    ("Basketball",  "post_up"):              BASKETBALL_POST_UP,
    ("Basketball",  "fast_break"):           BASKETBALL_FAST_BREAK,
    ("Rugby Union", "ball_carrier_contact"): RUGBY_BALL_CARRIER_HIT,
    ("Rugby Union", "breakdown_contest"):    RUGBY_BREAKDOWN,
    ("Rugby Union", "lineout_call"):         RUGBY_LINEOUT,
    ("Cricket",     "batting_delivery"):     CRICKET_BATTING_DELIVERY,
}


def register_extended_configs() -> None:
    """Merge EXTENDED_CONFIGS into the global ALL_MOMENT_CONFIGS registry.

    Call this once at application startup, after importing this module.
    """
    ALL_MOMENT_CONFIGS.update(EXTENDED_CONFIGS)
