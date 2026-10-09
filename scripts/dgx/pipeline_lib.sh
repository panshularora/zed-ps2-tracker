# shellcheck shell=bash
# Stage engine for `run.sh all` (scripts/dgx/run_all.sh). Source it; don't run it.
#
# A pipeline is an ordered list of stages registered with `pipe_add NAME KIND DESCRIPTION`.
# The caller defines `pipe_exec NAME`, which runs one stage. KIND is
#   once    runs once; success writes the done-marker $PIPE_STATE/done/NAME and a re-run skips it
#   always  runs on every invocation (cheap checks); never writes a marker
# Rules:
#   * fail-fast: each stage runs in a subshell with `set -Eeuo pipefail`; the first failing
#     command stops the pipeline, names the stage, command and line, and marks STATUS FAILED;
#   * a stage that calls `pipe_skip "reason"` ends as "skipped" with no marker (it is retried on
#     the next run, e.g. SOMPT22 once its manual download is there);
#   * when a `once` stage really runs, the markers of every LATER stage are removed, so nothing
#     downstream is stale (e.g. new caches -> ingest, phase 3 and the results tar are redone);
#   * PIPE_FORCE_ALL=1 or PIPE_FORCE=" a b " re-runs those stages even if marked done;
#   * PIPE_DRY_RUN=1 prints what would run, and touches nothing.
# STATUS ($PIPE_STATUS, default $ZED_ROOT/STATUS) is rewritten after every change: `cat` it.

: "${PIPE_STATE:=$ZED_ROOT/run_all}"
: "${PIPE_STATUS:=$ZED_ROOT/STATUS}"
: "${PIPE_DRY_RUN:=0}"
: "${PIPE_FORCE_ALL:=0}"
: "${PIPE_FORCE:=}"
: "${PIPE_TITLE:=run.sh all}"
: "${PIPE_LOG:=}"
PIPE_STAGES=()
declare -A PIPE_KIND=() PIPE_DESC=() PIPE_STATE_OF=() PIPE_INFO=()
PIPE_OVERALL="PENDING"
PIPE_CURRENT=""

pipe_now() { date '+%Y-%m-%d %H:%M:%S %Z'; }
pipe_dur() { local s=$1; printf '%dh%02dm%02ds' $((s / 3600)) $((s % 3600 / 60)) $((s % 60)); }

pipe_add() {   # NAME KIND DESCRIPTION
  [[ "$2" == once || "$2" == always ]] || { echo "pipe_add $1: bad kind '$2'" >&2; return 1; }
  [[ -z "${PIPE_KIND[$1]:-}" ]] || { echo "pipe_add: duplicate stage '$1'" >&2; return 1; }
  PIPE_STAGES+=("$1"); PIPE_KIND[$1]="$2"; PIPE_DESC[$1]="$3"; PIPE_STATE_OF[$1]="pending"
}

pipe_marker() { echo "$PIPE_STATE/done/$1"; }
pipe_is_done() { [[ -f "$(pipe_marker "$1")" ]]; }
pipe_forced() { [[ "$PIPE_FORCE_ALL" == 1 || " $PIPE_FORCE " == *" $1 "* ]]; }
pipe_has() { [[ -n "${PIPE_KIND[$1]:-}" ]]; }

# pipe_check_force : every name in PIPE_FORCE must be a registered stage
pipe_check_force() {
  local n
  for n in $PIPE_FORCE; do
    pipe_has "$n" || { echo "unknown stage in --force: '$n' (stages: ${PIPE_STAGES[*]})" >&2; return 1; }
  done
}

# Called from inside a stage: end it as "skipped" (no marker, retried next run)
pipe_skip() { mkdir -p "$PIPE_STATE"; printf '%s\n' "$*" > "$PIPE_STATE/.skip"; }

# Called from anywhere (stage subshells included): a line in STATUS "notes" + the log
pipe_note() {
  mkdir -p "$PIPE_STATE"
  printf '[%s] %s\n' "$(date '+%H:%M')" "$*" >> "$PIPE_STATE/notes.txt"
  printf '\033[1;32m[%s] NOTE:\033[0m %s\n' "$(date +%H:%M:%S)" "$*"
}

pipe_write_status() {
  [[ "$PIPE_DRY_RUN" == 1 ]] && return 0
  mkdir -p "$PIPE_STATE" "$(dirname "$PIPE_STATUS")"
  local tmp="$PIPE_STATUS.tmp.$$" n i=0
  {
    printf '%s   state: %s   (updated %s)\n' "$PIPE_TITLE" "$PIPE_OVERALL" "$(pipe_now)"
    [[ -n "$PIPE_CURRENT" ]] && printf 'current stage: %s\n' "$PIPE_CURRENT"
    [[ -n "$PIPE_LOG" ]] && printf 'log: %s\n' "$PIPE_LOG"
    printf '\n'
    for n in "${PIPE_STAGES[@]}"; do
      i=$((i + 1))
      printf ' %2d. %-10s %-22s %s\n' "$i" "[${PIPE_STATE_OF[$n]}]" "$n" "${PIPE_INFO[$n]:-}"
    done
    if [[ -s "$PIPE_STATE/notes.txt" ]]; then
      printf '\nnotes:\n'; sed 's/^/  /' "$PIPE_STATE/notes.txt"
    fi
  } > "$tmp"
  mv -f "$tmp" "$PIPE_STATUS"
}

pipe_dry_run() {
  local n i=0 action upstream=0
  printf 'DRY RUN: %s would run these stages in order (nothing is executed):\n\n' "$PIPE_TITLE"
  for n in "${PIPE_STAGES[@]}"; do
    i=$((i + 1))
    if [[ "${PIPE_KIND[$n]}" == always ]]; then
      action="run (every time)"
    elif pipe_forced "$n"; then
      action="RUN (forced)"; upstream=1
    elif pipe_is_done "$n" && [[ $upstream == 0 ]]; then
      action="skip (done $(head -1 "$(pipe_marker "$n")" 2>/dev/null))"
    elif pipe_is_done "$n"; then
      action="RUN (an earlier stage re-runs)"
    else
      action="RUN"; upstream=1
    fi
    printf ' %2d. %-22s %-7s %-34s %s\n' "$i" "$n" "${PIPE_KIND[$n]}" "$action" "${PIPE_DESC[$n]}"
  done
  printf '\nSTATUS file: %s   done-markers: %s/done/\n' "$PIPE_STATUS" "$PIPE_STATE"
}

pipe__on_exit() {
  local rc=$?
  if [[ "$PIPE_OVERALL" == RUNNING ]]; then
    PIPE_OVERALL="INTERRUPTED (exit $rc)"
    [[ -n "$PIPE_CURRENT" ]] && PIPE_STATE_OF[$PIPE_CURRENT]="stopped"
    pipe_write_status
  fi
}

# pipe_run : run every registered stage in order. Returns non-zero on the first failure.
pipe_run() {
  local n j rc t0 dt reason later
  pipe_check_force || return 2
  if [[ "$PIPE_DRY_RUN" == 1 ]]; then pipe_dry_run; return 0; fi
  mkdir -p "$PIPE_STATE/done"
  : > "$PIPE_STATE/notes.txt"
  trap pipe__on_exit EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  PIPE_OVERALL="RUNNING"
  for n in "${PIPE_STAGES[@]}"; do
    if pipe_is_done "$n"; then PIPE_STATE_OF[$n]="done"; PIPE_INFO[$n]="(earlier run: $(head -1 "$(pipe_marker "$n")"))"; fi
  done
  for n in "${PIPE_STAGES[@]}"; do
    if [[ "${PIPE_KIND[$n]}" == once ]]; then
      if pipe_forced "$n"; then
        rm -f "$(pipe_marker "$n")"
      elif pipe_is_done "$n"; then
        printf '\033[1;34m[%s]\033[0m stage %s: done earlier, skipped (--force=%s to redo)\n' \
          "$(date +%H:%M:%S)" "$n" "$n"
        continue
      fi
    fi
    PIPE_CURRENT="$n"; PIPE_STATE_OF[$n]="running"; PIPE_INFO[$n]="since $(date '+%H:%M:%S')"
    pipe_write_status
    printf '\n\033[1;35m[%s] ===== stage %s: %s =====\033[0m\n' "$(date +%H:%M:%S)" "$n" "${PIPE_DESC[$n]}"
    rm -f "$PIPE_STATE/.skip"
    t0=$(date +%s)
    set +e
    (
      set -Eeuo pipefail
      trap 'rc=$?; printf "\033[1;31m[%s] ERROR in stage %s: \"%s\" exited %s (line %s)\033[0m\n" "$(date +%H:%M:%S)" "'"$n"'" "$BASH_COMMAND" "$rc" "$LINENO" >&2' ERR
      pipe_exec "$n"
    )
    rc=$?
    set -e
    dt=$(( $(date +%s) - t0 ))
    if [[ $rc -ne 0 ]]; then
      PIPE_STATE_OF[$n]="FAILED"; PIPE_INFO[$n]="exit $rc after $(pipe_dur "$dt")"
      PIPE_OVERALL="FAILED at stage $n"
      PIPE_CURRENT=""
      pipe_write_status
      printf '\n\033[1;31mFAILED at stage %s (exit %s).\033[0m Fix the cause above, then re-run the same command:\n' "$n" "$rc" >&2
      printf 'finished stages are skipped. Status: cat %s   Log: %s\n' "$PIPE_STATUS" "${PIPE_LOG:-(none)}" >&2
      return "$rc"
    fi
    if [[ -f "$PIPE_STATE/.skip" ]]; then
      reason="$(cat "$PIPE_STATE/.skip")"; rm -f "$PIPE_STATE/.skip"
      PIPE_STATE_OF[$n]="skipped"; PIPE_INFO[$n]="$reason"
      printf '\033[1;33m[%s] stage %s skipped: %s\033[0m\n' "$(date +%H:%M:%S)" "$n" "$reason"
    else
      PIPE_STATE_OF[$n]="done"; PIPE_INFO[$n]="$(pipe_dur "$dt")"
      if [[ "${PIPE_KIND[$n]}" == once ]]; then
        printf '%s\n%s\n' "$(pipe_now)" "${PIPE_MARK_INFO:-}" > "$(pipe_marker "$n")"
        later=0
        for j in "${PIPE_STAGES[@]}"; do   # every later stage is now stale
          if [[ $later == 1 ]] && pipe_is_done "$j"; then
            rm -f "$(pipe_marker "$j")"; PIPE_STATE_OF[$j]="pending"; PIPE_INFO[$j]="(redo: an earlier stage re-ran)"
          fi
          [[ "$j" == "$n" ]] && later=1
        done
      fi
    fi
    PIPE_CURRENT=""
    pipe_write_status
  done
  PIPE_OVERALL="DONE"
  pipe_write_status
  return 0
}
