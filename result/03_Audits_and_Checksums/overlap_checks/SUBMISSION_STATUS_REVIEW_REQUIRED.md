# Submission status

## Status: REVIEW REQUIRED BEFORE SENDING

The technical result files have been filtered into one canonical package.
Outdated/conflicting summaries are intentionally excluded.

However, the supplied final 870-image cross-domain audit reports:

- Benchmark test images: 17,400
- Cross-domain images used: 870
- Exact SHA256 overlap pairs between benchmark test and cross-domain: **55**

This is material because the manuscript currently describes the cross-domain
set as independent. The overlap must be reconciled or explicitly disclosed
before the package is sent to the editor.

Also note:
- The raw cross-domain folder contained 871 images; the final evaluation uses
  exactly 870 (30/class). The excluded file and selection audit are included.
- Detection benchmark/cross-domain mAP was not rerun from these class-folder
  datasets because verified bounding-box ground truth was unavailable.
  The package includes only a clearly labeled metric-consistent mAP50
  recalculation from the values already reported in Tables 3 and 4.
- Model weights are not bundled in this ZIP. Checkpoint filenames, sizes, and
  SHA256 hashes are included. If the editor requests weights, provide them via
  a stable release/LFS link and record the commit/release identifier.
