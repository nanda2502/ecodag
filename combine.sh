#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./combine.sh <postfix> [output_dir]

Combines per-row result CSVs for a postfix, keeping one header.

Inputs:
  output/results_<postfix>_*.csv
  output/results_avg_<postfix>_*.csv

Outputs:
  output/results_<postfix>_combined.csv
  output/results_avg_<postfix>_combined.csv
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit $([[ $# -ge 1 && "${1:-}" =~ ^(-h|--help)$ ]] && echo 0 || echo 1)
fi

postfix="$1"
outdir="${2:-output}"

combine_group() {
  local prefix="$1"
  local destination="$outdir/${prefix}_${postfix}_combined.csv"
  local files=()

  shopt -s nullglob
  files=("$outdir/${prefix}_${postfix}_"*.csv)
  shopt -u nullglob

  local filtered=()
  local file
  for file in "${files[@]}"; do
    [[ "$file" == "$destination" ]] && continue
    [[ "$file" == *_combined.csv ]] && continue
    filtered+=("$file")
  done

  if (( ${#filtered[@]} == 0 )); then
    echo "No files found for $outdir/${prefix}_${postfix}_*.csv" >&2
    return 0
  fi

  printf '%s\n' "${filtered[@]}" | sort > "$destination.files"
  local first=1
  : > "$destination"
  while IFS= read -r file; do
    file="${file%$'\r'}"
    if (( first )); then
      cat "$file" >> "$destination"
      first=0
    else
      tail -n +2 "$file" >> "$destination"
    fi
  done < "$destination.files"
  rm -f "$destination.files"

  echo "Wrote $destination from ${#filtered[@]} files"
}

combine_group "results"
combine_group "results_avg"
