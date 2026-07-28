#!/usr/bin/env bash
# Fetch the canonical fixtures. They are gitignored (12 MB) but every notebook,
# R driver and parity test expects them here.
#
#   bash data/download_fixtures.sh
#
# Source: the upstream Spatial EcoTyper vignette data server. Melanoma1 is the
# fixture for upstream Tutorial 1 (500 genes x 27,907 cells, MERSCOPE, 9
# non-malignant cell types); CRC2 is the second sample for Tutorial 2's
# integration (38,080 cells) and is the held-out fixture in data/manifest.yaml.

set -euo pipefail
cd "$(dirname "$0")"

BASE="https://spatialecotyper.stanford.edu/inc/inc.public.vignettes.php?file="

for f in Melanoma1_subset_counts.tsv.gz Melanoma1_subset_scmeta.tsv \
         CRC2_subset_counts.tsv.gz CRC2_subset_scmeta.tsv; do
  if [ -s "$f" ]; then
    echo "have  $f"
  else
    echo "fetch $f"
    curl -sSL --max-time 900 -o "$f" "${BASE}${f}"
  fi
done

echo
echo "Fixtures ready:"
ls -la Melanoma1_subset_* CRC2_subset_*
