"""
How each side used its clock in Lichess games: did a player start thinking much less
later in the game (rushing), and was that because its clock was running low?

    uv run python Engine/clockreport.py                         # slackfish-bot's last 10 games
    uv run python Engine/clockreport.py --max 40 --perf blitz    # last 40 blitz games
    uv run python Engine/clockreport.py rZTHS0VQ https://lichess.org/T6hlGD7n   # given games
    uv run python Engine/clockreport.py --max 20 --plot clocks/  # + a clock chart per game

Games given by id or link need no login. A user's recent games do: Lichess only exports
them with a token, read like Engine/lichess.py from LICHESS_TOKEN or .lichess_token at
the repo root (any token works; nothing is posted). For each move it takes the clock the mover
had left, and the time the move took (previous clock + increment - clock). Each side's
moves are then split into:

  early   its first --window moves from move 10 on (past the opening book, while both
          sides still have most of their time): the player's normal thinking time
  late    its last --window moves

and measured:

  early/late    median seconds per move in each window
  ratio         late / early. Under --rush (default 0.4): the player was thinking much
                less by the end ("rushed")
  rush from     the move from which the median of all its remaining moves stays under
                --rush x early, and the clock it had then
  share         median fraction of its remaining clock spent per move, early and late.
                A player that budgets a fixed share of its clock (as slackfish-bot does:
                1/12, 1/20 under 3 minutes) slows down as its clock falls while its
                share stays level; a share that falls too means it sped up beyond that
  instant       how many late-window moves took under 0.3 s: a pondering bot's replies
                to a predicted move (slackfish-bot plays those at once) or premoves
  low clock     lowest clock it had, and "trouble" when that was under 10% of the
                starting time (or 15 s)

Verdicts for a player that rushed:
  clock-limited    it thought less because its clock was low (what this report is for):
                   the rush began with under a quarter of the starting time left, or it
                   kept spending the same or a bigger share of a shrinking clock (its
                   share didn't fall below 0.8x the early share)
  instant replies  not clock-limited, and at least half its late moves were instant
                   (pondering or premoves, not hurrying)
  by choice        it had time and its share fell too: an easy or forced position, or
                   an engine that plays simple endgames fast The summary counts these per
opponent, with slackfish-bot's score in the games where the opponent was clock-limited.
"""
import argparse
import json
import os
import re
import statistics
import sys

import requests

API = "https://lichess.org"
ME = "slackfish-bot"
INSTANT = 0.3   # seconds: a move this fast was pondered or premoved, not thought about
TOKEN_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".lichess_token")


def token():
    if os.environ.get("LICHESS_TOKEN"):
        return os.environ["LICHESS_TOKEN"].strip()
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            return f.read().strip()
    return None


def game_id(text):
    m = re.search(r"(?:lichess\.org/)?([A-Za-z0-9]{8})(?:[A-Za-z0-9]{4})?(?:/(?:white|black))?/?$", text.strip())
    if not m:
        raise SystemExit(f"not a Lichess game id or link: {text}")
    return m.group(1)


def fetch_games(args):
    params = {"clocks": "true", "moves": "true", "evals": "false", "opening": "false"}
    headers = {"Accept": "application/x-ndjson"}
    if args.games:
        r = requests.post(f"{API}/api/games/export/_ids", params=params, headers=headers,
                          data=",".join(game_id(g) for g in args.games), timeout=60)
    else:
        params["max"] = args.max
        if args.perf:
            params["perfType"] = args.perf
        tok = token()
        if tok is None:
            raise SystemExit("Lichess only exports a user's games with a token: set LICHESS_TOKEN or create "
                             f"{TOKEN_FILE}, or pass game ids/links instead")
        headers["Authorization"] = f"Bearer {tok}"
        r = requests.get(f"{API}/api/games/user/{args.user}", params=params, headers=headers, timeout=60)
    r.raise_for_status()
    games = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    return [g for g in games if g.get("clocks") and g.get("clock")]


def name(game, color):
    p = game["players"][color]
    return p.get("user", {}).get("name") or (f"AI level {p['aiLevel']}" if "aiLevel" in p else "?")


def side_moves(game, color):
    """[(move number, seconds left before the move, seconds the move took)] for one side.
    Each side's first move is skipped: Lichess starts its clock after it."""
    clocks = [c / 100 for c in game["clocks"]]
    inc = game["clock"]["increment"]
    start = 0 if color == "white" else 1
    plies = range(start, len(clocks), 2)
    out = []
    prev = None
    for ply in plies:
        if prev is not None:
            took = max(0.0, prev + inc - clocks[ply])
            out.append((ply // 2 + 1, prev, took))
        prev = clocks[ply]
    return out


def analyse(game, color, args):
    moves = side_moves(game, color)
    initial = game["clock"]["initial"]
    usable = [m for m in moves if m[0] >= args.from_move]
    res = {"player": name(game, color), "color": color, "moves": len(moves),
           "end clock": game["clocks"][-1 if (len(game["clocks"]) % 2 == 1) == (color == "white") else -2] / 100,
           "low clock": min((m[1] for m in moves), default=initial)}
    res["trouble"] = res["low clock"] < max(15.0, 0.1 * initial)
    if len(usable) < 2 * args.window:
        res["status"] = "too short"
        return res
    early, late = usable[:args.window], usable[-args.window:]
    e = statistics.median(m[2] for m in early)
    lt = statistics.median(m[2] for m in late)
    res.update(early=e, late=lt, ratio=lt / e if e > 0 else None,
               share_early=statistics.median(m[2] / m[1] for m in early if m[1] > 0),
               share_late=statistics.median(m[2] / m[1] for m in late if m[1] > 0),
               instant=sum(m[2] < INSTANT for m in late))
    res["rush from"] = None
    for i, (num, left, _) in enumerate(usable):
        rest = [m[2] for m in usable[i:]]
        if len(rest) >= args.window // 2 and statistics.median(rest) < args.rush * e:
            res["rush from"] = (num, left)
            break
    if res["ratio"] is not None and res["ratio"] < args.rush and res["rush from"] is not None:
        if res["rush from"][1] < 0.25 * initial or res["share_late"] >= 0.8 * res["share_early"]:
            res["status"] = "rushed, clock-limited"
        elif res["instant"] * 2 >= len(late):
            res["status"] = "rushed, instant replies"
        else:
            res["status"] = "rushed, by choice"
    else:
        res["status"] = "steady"
    return res


def score(game, color):
    w = game.get("winner")
    return 0.5 if w is None else 1.0 if w == color else 0.0


def fmt_clock(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


def plot(game, results, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for color, res in results.items():
        moves = side_moves(game, color)
        nums = [m[0] for m in moves]
        line, = ax1.plot(nums, [m[1] for m in moves], label=f"{res['player']} ({color})")
        ax2.plot(nums, [m[2] for m in moves], color=line.get_color(), marker=".", linewidth=0.8)
        if res.get("rush from"):
            for ax in (ax1, ax2):
                ax.axvline(res["rush from"][0], color=line.get_color(), linestyle="--", linewidth=0.8)
    ax1.set_ylabel("clock left (s)")
    ax2.set_ylabel("seconds per move")
    ax2.set_xlabel("move")
    ax1.legend()
    c = game["clock"]
    ax1.set_title(f"{name(game, 'white')} vs {name(game, 'black')}, {c['initial'] // 60}+{c['increment']}, "
                  f"{game['status']} (dashed: rush from)")
    fig.tight_layout()
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{game['id']}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("games", nargs="*", help="game ids or links (default: --user's recent games)")
    p.add_argument("--user", default=ME, help=f"whose games to fetch (default {ME})")
    p.add_argument("--max", type=int, default=10, help="recent games to fetch (default 10)")
    p.add_argument("--perf", choices=["bullet", "blitz", "rapid", "classical"], help="only this time control")
    p.add_argument("--window", type=int, default=10, help="moves in the early and late windows (default 10)")
    p.add_argument("--from-move", type=int, default=10, help="first move of the early window (default 10)")
    p.add_argument("--rush", type=float, default=0.4, help="late/early ratio that counts as rushing (default 0.4)")
    p.add_argument("--plot", metavar="DIR", help="save a clock chart per game here")
    args = p.parse_args()

    games = fetch_games(args)
    if not games:
        raise SystemExit("no games with clocks found")
    me = (args.user if not args.games else ME).lower()
    summary = {}
    for g in reversed(games):   # oldest first
        results = {c: analyse(g, c, args) for c in ("white", "black")}
        c = g["clock"]
        print(f"\nhttps://lichess.org/{g['id']}  {name(g, 'white')} vs {name(g, 'black')}  "
              f"{c['initial'] // 60}+{c['increment']}  {g['status']}, "
              f"{'draw' if g.get('winner') is None else g['winner'] + ' won'}")
        print(f"  {'player':<20} {'moves':>5} {'early':>6} {'late':>6} {'ratio':>6} {'share':>12} {'instant':>7} "
              f"{'rush from':>15} {'low clock':>10} {'end':>6}  verdict")
        for color, r in results.items():
            if "early" in r:
                share = f"{100 * r['share_early']:.0f}%->{100 * r['share_late']:.0f}%"
                rush = f"move {r['rush from'][0]} @{fmt_clock(r['rush from'][1])}" if r["rush from"] else "-"
                inst = f"{r['instant']}/{args.window}"
                cols = f"{r['early']:5.1f}s {r['late']:5.1f}s {r['ratio']:6.2f} {share:>12} {inst:>7} {rush:>15}"
            else:
                cols = f"{'':>6} {'':>6} {'':>6} {'':>12} {'':>7} {'':>15}"
            trouble = " (trouble)" if r["trouble"] else ""
            print(f"  {r['player'][:20]:<20} {r['moves']:>5} {cols} {fmt_clock(r['low clock']):>10} "
                  f"{fmt_clock(r['end clock']):>6}  {r['status']}{trouble}")
        if args.plot:
            print(f"  chart: {plot(g, results, args.plot)}")
        mine = [col for col, r in results.items() if r["player"].lower() == me]
        if mine:
            opp = results["black" if mine[0] == "white" else "white"]
            s = summary.setdefault(opp["player"], {"games": 0, "score": 0.0, "rushed clock": 0, "rushed other": 0,
                                                    "trouble": 0, "score when rushed": []})
            s["games"] += 1
            s["score"] += score(g, mine[0])
            s["trouble"] += opp["trouble"]
            if opp["status"].startswith("rushed"):
                s["rushed clock" if "clock" in opp["status"] else "rushed other"] += 1
                if "clock" in opp["status"]:
                    s["score when rushed"].append(score(g, mine[0]))
    if summary:
        print(f"\nOpponents of {me} ({sum(s['games'] for s in summary.values())} games):")
        print(f"  {'opponent':<20} {'games':>5} {'score':>6} {'rushed (clock-limited)':>23} {'rushed (other)':>15} "
              f"{'clock trouble':>14} {'our score when clock-limited':>29}")
        for opp, s in sorted(summary.items(), key=lambda kv: -kv[1]["games"]):
            when = s["score when rushed"]
            print(f"  {opp[:20]:<20} {s['games']:>5} {s['score']:>6g} {s['rushed clock']:>23} {s['rushed other']:>15} "
                  f"{s['trouble']:>14} {(f'{sum(when):g}/{len(when)}' if when else '-'):>29}")


if __name__ == "__main__":
    main()
