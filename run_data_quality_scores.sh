#!/usr/bin/env bash
set -u
set -o pipefail

usage() {
  echo "Usage: $0 <data_directory> [--json] [--output-dir <directory>]"
  echo
  echo "Runs data_quality_score.py for every .csv file under <data_directory>."
  echo
  echo "Options:"
  echo "  --json                 Print each result as JSON."
  echo "  --output-dir <dir>     Write one JSON report per CSV into <dir>."
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scorer="${script_dir}/data_quality_score.py"

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

data_dir="$1"
shift

json_flag=""
output_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      json_flag="--json"
      shift
      ;;
    --output-dir)
      if [[ $# -lt 2 ]]; then
        echo "Error: --output-dir requires a directory path." >&2
        exit 2
      fi
      output_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'." >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -f "$scorer" ]]; then
  echo "Error: data quality scorer not found at: $scorer" >&2
  exit 2
fi

if [[ ! -d "$data_dir" ]]; then
  echo "Error: data directory not found: $data_dir" >&2
  exit 2
fi

if [[ -n "$output_dir" ]]; then
  mkdir -p "$output_dir"
fi

csv_count=0
pass_count=0
fail_count=0

while IFS= read -r -d '' csv_file; do
  csv_count=$((csv_count + 1))

  if [[ -n "$output_dir" ]]; then
    relative_path="${csv_file#"$data_dir"/}"
    report_name="${relative_path//\//__}.quality.json"
    report_path="${output_dir%/}/${report_name}"
    python3 "$scorer" "$csv_file" --json --output "$report_path"
  elif [[ -n "$json_flag" ]]; then
    python3 "$scorer" "$csv_file" --json
  else
    echo "============================================================"
    python3 "$scorer" "$csv_file"
    echo
  fi

  status=$?
  if [[ $status -eq 0 ]]; then
    pass_count=$((pass_count + 1))
  else
    fail_count=$((fail_count + 1))
  fi
done < <(find "$data_dir" -type f -name '*.csv' -print0 | sort -z)

if [[ $csv_count -eq 0 ]]; then
  echo "No CSV files found under: $data_dir" >&2
  exit 1
fi

echo "Scored $csv_count CSV file(s): $pass_count passed, $fail_count failed."

if [[ $fail_count -gt 0 ]]; then
  exit 1
fi

exit 0
