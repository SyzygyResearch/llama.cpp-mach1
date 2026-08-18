#!/usr/bin/env bash
# Package HEAD as a rev-suffixed tarball, upload it, and point the mounted
# bench script's SRC_REV at it. Rev-unique paths make volume-cache staleness
# fail loudly (missing file) instead of silently building old code.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
rev=$(git rev-parse --short HEAD)
sed -i "s/^SRC_REV = .*/SRC_REV = \"$rev\"/" benches/modal/bench_h200.py
tmp=$(mktemp -d)
git archive HEAD | gzip -1 > "$tmp/tree.tar.gz"
python3 -m modal volume put mach1-build-cache "$tmp/tree.tar.gz" "/src/tree-$rev.tar.gz" --force
rm -rf "$tmp"
echo "uploaded /src/tree-$rev.tar.gz; bench_h200.py SRC_REV=$rev (working tree)"
