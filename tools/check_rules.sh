#!/usr/bin/env bash
#
# Fail when a file under .claude/rules/ does not declare the paths it is scoped to.
#
# Those files are loaded automatically for whoever is working in the repository, selected by the
# `paths:` glob in each file's frontmatter so that a file loads only while the code it describes is
# being read or edited. A file with NO `paths:` is not scoped to nothing -- it is scoped to
# EVERYTHING: it loads for every session whatever is being worked on.
#
# That inversion is silent from both ends. The file looks inert, and the sessions paying for it
# report only that they began with less room than they should have. Three reference-only files --
# two investigation logs and a ledger of refuted directions -- sat there with the frontmatter left
# off, each opening with the words "this file never auto-loads", and all three loaded into every
# session in the repository: some 285 KB, very roughly seventy thousand tokens, before the first
# tool call. They now live in .claude/notes/, which is not scanned.
#
# So: a file under .claude/rules/ must carry `paths:`. Reference-only material that is meant to be
# read deliberately belongs in .claude/notes/ instead.
#
# Usage
#   tools/check_rules.sh [dir]    dir defaults to <repo top>/.claude/rules
#
# Exits 0 when every file declares its scope, 1 otherwise. Unlike tools/check_hooks.sh this is a
# gate, not a warning: the condition is mechanical, unambiguous, and expensive to leave in place.

set -uo pipefail

dir="${1:-}"
if [ -z "$dir" ]; then
  top=$(git rev-parse --show-toplevel 2>/dev/null) || {
    printf 'rules: not inside a work tree and no directory given.\n' >&2
    exit 1
  }
  dir="$top/.claude/rules"
fi

if [ ! -d "$dir" ]; then
  printf 'rules: no such directory: %s\n' "$dir" >&2
  exit 1
fi

status=0
checked=0

for f in "$dir"/*.md; do
  # A directory holding no .md files leaves the glob unexpanded; that is not a violation.
  [ -e "$f" ] || continue
  checked=$((checked + 1))

  # `paths:` counts only inside the leading `---` frontmatter block: a mention in the prose below
  # it is documentation, not declaration, and several rules files legitimately discuss the key.
  if [ "$(head -1 "$f")" != "---" ]; then
    printf 'rules: %s has no frontmatter block.\n' "${f#"$dir"/}" >&2
  elif ! sed -n '2,/^---$/p' "$f" | grep -q '^paths:'; then
    printf 'rules: %s has frontmatter but no `paths:` key.\n' "${f#"$dir"/}" >&2
  else
    continue
  fi

  printf '       Without it the file loads for EVERY session regardless of what is being worked\n' >&2
  printf '       on -- the opposite of being scoped to nothing. Give it the glob it applies to,\n' >&2
  printf '       or move it to .claude/notes/ if it is reference-only material.\n' >&2
  status=1
done

# Guard the guard: a renamed directory or a broken glob would otherwise pass by checking nothing,
# which looks exactly like a clean tree.
if [ "$checked" -eq 0 ]; then
  printf 'rules: examined no files in %s -- this check has stopped seeing anything.\n' "$dir" >&2
  exit 1
fi

exit $status
