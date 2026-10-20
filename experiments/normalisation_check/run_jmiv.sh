#!/usr/bin/env bash
# Rerun the experiments/pairs_and_rules analysis scripts, unchanged, on the
# MPEG-7 features of each normalisation variant written by extract.py.
#
# Each variant directory gets a copy of the scripts and their fixed inputs, so
# the scripts read the CSVs next to them exactly as in the original run. Only
# MPEG-7 is recomputed: the Animal2000 and SwedishLeaves inputs are symlinks to
# the MPEG-7 files purely so the scripts run to completion -- every
# non-MPEG-7 line in the logs is meaningless and must be ignored.
#
# Usage: run_jmiv.sh PYTHON [OUT]   (PYTHON needs numpy, scikit-learn, scipy)
set -euo pipefail
PY=${1:?python with scikit-learn}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${2:-$HERE/out}
SRC=$HERE/../pairs_and_rules
REPO=$(cd "$HERE/../.." && pwd)

SCRIPTS=${SCRIPTS:-"analyze disjmeasure disjdim disjhead residue classdisjoint classdisjoint2
         classdisjoint_general rfcheck asymmetry_test generalpair_test aperture_confound
         maketables make_tab_replacement make_tab_headline makefigdata
         fourier_test fourier_variants classdisjoint_fourier make_tab_fourier make_tab_count"}

for v in code generator; do
  d=$OUT/$v
  cp "$SRC"/*.py "$SRC"/slotcols.pkl "$SRC"/wide_pairs.json "$d"/
  for ds in animal2000 swedishleaves; do
    ln -sf mpeg7_full_features.csv "$d/${ds}_features.csv"
    for k in disjunctive asymmetry generalpairs widepairs fourier; do
      ln -sf "mpeg7_$k.csv" "$d/${ds}_$k.csv"
    done
  done
  mkdir -p "$d/logs" "$d/tab" "$d/fig"
  rm -f "$d/logs/_failures.txt"
  [ -n "${KEEP_RF:-}" ] || rm -f "$d/rf_results.csv"
  (
    cd "$d"
    for s in $SCRIPTS; do
      # fourier_variants re-reads the raw images, which exist for MPEG-7 only
      case $s in fourier_variants) a=MPEG-7 ;; *) a= ;; esac
      PYTHONPATH=$REPO QCONV_DATA=$SRC/data QCONV_TAB=$d/tab QCONV_FIG=$d/fig "$PY" "$s.py" $a > "logs/$s.log" 2>&1 \
        || echo "$v: $s exited $?" >> logs/_failures.txt
    done
  ) &
done
wait
echo done
