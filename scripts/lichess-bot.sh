#!/usr/bin/env bash
# Run slackfish-bot (Engine/lichess.py) in the background.
#
#   scripts/lichess-bot.sh start [EXTRA ARGS]   # go online and accept challenges
#   scripts/lichess-bot.sh blitz [EXTRA ARGS]   # also challenge bots: 10 rated 3+2 games (alias: series)
#   scripts/lichess-bot.sh bullet [EXTRA ARGS]  # same with 10 rated 1+1 games, tuned for bullet
#   scripts/lichess-bot.sh stop [--now]         # stop once no game is in progress (--now: right away)
#   scripts/lichess-bot.sh restart [ARGS]       # stop between games, then start with ARGS
#   scripts/lichess-bot.sh status               # running? online? current game?
#   scripts/lichess-bot.sh log                  # follow the log
#
# EXTRA ARGS go to `lichess.py play`, after the defaults below, so they override
# them, e.g. `blitz --games 6`, `bullet --clock 2+1` or `start --bot t1-3.25m:3`.
# Stopping mid-game loses that game on time unless the bot is started again soon
# (a new start resumes games in progress), so stop waits for the game to end.
# Needs the bot's token in .lichess_token (see Engine/lichess.py).
#
# ---------------------------------------------------------------------------
# Notes for an AI assistant asked to operate the bot
# ---------------------------------------------------------------------------
# What this is: slackfish-bot is a Lichess BOT account (https://lichess.org/@/slackfish-bot)
# driven by Engine/lichess.py on this machine. It is only online while that
# process runs. One process = one bot; never start a second one (start refuses
# if the pid in logs/lichess.pid is alive).
#
# Always check first:   scripts/lichess-bot.sh status
#   "running (pid N)" / "not running"   whether the process is up
#   "online":true                       Lichess sees it (takes ~10 s after start)
#   "playing: https://lichess.org/ID"   a game is in progress right now
#
# Rules that matter:
#   - Don't use `stop --now` or `kill` while a game is in progress: the bot stops
#     moving and loses on time. Plain `stop` / `restart` wait for the game to end
#     (they poll every 10 s, so they can take minutes; run them in the background
#     if you need to keep working). Code changes to Engine/*.py only take effect
#     after a restart.
#   - Never print, log or paste the contents of .lichess_token.
#   - A finished blitz/bullet run (preset) does NOT take the bot offline; it
#     keeps accepting challenges until stopped.
#   - Other bots are limited by Lichess to 100 bot games a day; "played 100
#     games against other bots today" errors in the log are normal, the bot
#     skips that opponent. So are "declined" lines.
#
# Common requests -> command:
#   "put the bot online"                  scripts/lichess-bot.sh start
#   "take it offline"                     scripts/lichess-bot.sh stop
#   "play N blitz games vs bots"          scripts/lichess-bot.sh stop && scripts/lichess-bot.sh blitz --games N
#   "play N bullet games vs bots"         scripts/lichess-bot.sh stop && scripts/lichess-bot.sh bullet --games N
#                                         (restart can't start a preset; both use OPPONENTS below)
#   "use another model / depth"           scripts/lichess-bot.sh restart --bot NAME:DEPTH
#                                         (models: Engine/bots.py BOTS; e.g. t1-3.25m is weaker, for beginners)
#   "casual games only"                   scripts/lichess-bot.sh restart --casual-only
#   "play Lichess's Stockfish"            scripts/lichess-bot.sh restart --stockfish 1,2,3 --games 3
#   "what's it doing / how did it go"     scripts/lichess-bot.sh status, then read logs/lichess.out
#   "what's its rating"                   curl -s https://lichess.org/api/user/slackfish-bot
#                                         (JSON: perfs.blitz.rating, perfs.bullet.rating; "prov": true = provisional)
# Every option: .venv/bin/python Engine/lichess.py play --help
#
# Bullet vs blitz settings (what the presets change, and how to adjust them):
#                 clock   think per move (budget)                     pondered reply played at once from
#   blitz         3+2     1/12 of clock + increment: ~17 s at 3:00    depth 5   (--bot t2-sb-wsd-lichessgames-6.97m:5)
#   bullet        1+1     1/20 of clock + increment: ~4 s at 1:00,    depth 4   (--bot t2-sb-wsd-lichessgames-6.97m:4)
#                         3 s at 0:40, 2 s at 0:20, 1.4 s at 0:08
#   - The 1/12 vs 1/20 is chosen automatically from each game's starting clock
#     (under 3 minutes = bullet), for ANY game, including friends' challenges in
#     plain `start` mode. So friends can challenge it to bullet without a preset.
#   - "bullet, but a different clock":   bullet --clock 2+1   (or 0.5+1 for 30 s; MINUTES+SECONDS)
#   - "bullet, casual (unrated)":        bullet --casual      (works for blitz too: blitz --casual)
#   - "think faster / slower":           add --moves-left N (higher = less time per move; 20 bullet, 12 blitz default).
#                                        Below ~15 in bullet risks losing on time; keep the increment (+1) in bullet.
#   - "play bullet more cautiously":     the :DEPTH in --bot: lower (e.g. :3) = more instant moves, saves clock.
#   - Check how it went: grep "of [0-9.]*s)" logs/lichess.out shows each move's time used vs budget;
#     "outoftime: lost" means it flagged (time settings too generous, or a connection problem in the log).
#
# Reading logs/lichess.out (one line per event):
#   [GAMEID] White vs NAME (blitz, rated): URL     a game started (bullet / blitz / rapid)
#   [GAMEID] 12.Nf3  (depth 5, pondered, 3.10s of 12.50s)
#        move played; search depth; "pondered" = it had thought about this
#        reply on the opponent's time; seconds used of the move's time budget.
#        "(book, ...)" = opening book move, no search.
#        "(tablebase win, mate in 7, ...)" = 7 pieces or fewer: the perfect move from
#        Lichess's endgame tablebase, no search (falls back to search if it's unreachable;
#        --no-tablebase turns it off). "tablebase draw" = it can't be won with best play.
#   [GAMEID] mate: won / resign: lost / draw: drew / outoftime: lost   game over
#   draw offered at -0.05 for us: accepting       draw offers (accepted below +0.10)
#   challenged NAME (3+2, rated), game 3/10        series (blitz/bullet preset) progress
#   SlackFish's score against the opponents it challenged: ...   end-of-series summary
# Older runs are appended to logs/lichess-history.out on each start.
set -euo pipefail
cd "$(dirname "$0")/.."   # run from the repo root, wherever the script is called from

LOG=logs/lichess.out
PIDFILE=logs/lichess.pid
# Opening book: Cerebellum (strong, downloaded separately, see README) or the bundled gm2001
BOOK=Cerebellum3Merge.bin
[ -f "Models/Books/$BOOK" ] || BOOK=gm2001.bin
# Model and depth: NAME:DEPTH. The search is timed with no depth limit; DEPTH is how
# deep a pondered reply must have been searched to be played without thinking more.
DEFAULTS=(--bot t2-sb-wsd-lichessgames-6.97m:5 --book "$BOOK" --book-plies 30)
# Bots for `series` to challenge in turn, all roughly 1600-2300 blitz. Any that decline
# twice or don't answer are skipped for the rest of the run. Edit freely.
OPPONENTS=CosetteBot,Hyperopic,turkjs,Bottios,Boris-Trapsky,kopyto_dev,stickshark99,uzenatch,matmoi,pi0w,sseh-c,TucuEngine,TopasBot,jangine,GarboBot
# Presets: games against OPPONENTS, then it stays online for challenges.
# Blitz uses the defaults: 1/12 of the clock + increment per move (about 17 s at 3:00),
# and a pondered reply searched to depth 5 is played at once.
BLITZ=(--challenge "$OPPONENTS" --games 10 --clock 3+2 --rated)
# Bullet: lichess.py already budgets 1/20 of the clock under 3 minutes (about 4 s at 1:00,
# so ~0.2 s of lag per move stays affordable); play pondered replies from depth 4, and
# depth 4 is what 3-4 s usually reaches. Friends' bullet challenges get the same timing.
BULLET=(--bot t2-sb-wsd-lichessgames-6.97m:4 --challenge "$OPPONENTS" --games 10 --clock 1+1 --rated)

# The bot's pid, if it's running (exit status 1 otherwise)
pid() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && cat "$PIDFILE"; }

# Ids of the bot's games in progress, space separated (empty if none), from Lichess
playing() {
    curl -s -H "Authorization: Bearer $(tr -d '[:space:]' < .lichess_token)" \
        https://lichess.org/api/account/playing |
        python3 -c 'import json,sys; print(" ".join(g["gameId"] for g in json.load(sys.stdin)["nowPlaying"]))'
}

start() {
    if p=$(pid); then echo "already running (pid $p); use restart"; exit 1; fi
    mkdir -p logs
    [ -f "$LOG" ] && cat "$LOG" >> logs/lichess-history.out   # keep the previous run's log
    # the venv's python directly, so the pid (and stop's signal) is the bot's own
    py=.venv/bin/python; [ -x "$py" ] || { uv sync -q; }
    nohup "$py" -u Engine/lichess.py play "${DEFAULTS[@]}" "$@" > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "started (pid $!), log: $LOG"
}

stop() {
    p=$(pid) || { echo "not running"; return 0; }
    if [ "${1:-}" != "--now" ]; then   # wait for games in progress, or they're lost on time
        while games=$(playing) && [ -n "$games" ]; do
            echo "waiting for game $games to end (stop --now to stop anyway)"
            sleep 10
        done
    fi
    kill "$p"
    while kill -0 "$p" 2>/dev/null; do sleep 0.5; done
    rm -f "$PIDFILE"
    echo "stopped"
}

case "${1:-}" in
    start) shift; start "$@" ;;
    blitz|series) shift; start "${BLITZ[@]}" "$@" ;;
    bullet) shift; start "${BULLET[@]}" "$@" ;;
    stop) shift; stop "$@" ;;
    restart) shift; stop; start "$@" ;;
    status)
        if p=$(pid); then echo "running (pid $p)"; else echo "not running"; fi
        curl -s "https://lichess.org/api/users/status?ids=slackfish-bot&withGameIds=true"; echo
        games=$(playing); [ -n "$games" ] && echo "playing: https://lichess.org/$games" || true ;;
    log) tail -f "$LOG" ;;
    *) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;   # usage: the first comment block
esac
