# Final Editor Evidence

This directory contains the canonical technical re-examination evidence package.

## Canonical Result Folders

```text
01_Benchmark_CrossDomain/
  Classification/
    IJIES_EDITOR_CLASSIFICATION_BENCHMARK_CROSSDOMAIN/
  Detection/
    IJIES_EDITOR_DETECTOR_BENCHMARK_CROSSDOMAIN/
02_RealTime_Video/
  IJIES_EDITOR_REALTIME_EVIDENCE/
03_Audits_and_Checksums/
  manifests/
  sha256/
  duplicate_checks/
  overlap_checks/
```

## Notes

- Classification evidence has exactly one canonical folder.
- Detection evidence has exactly one canonical folder.
- Real-time video evidence has exactly one canonical folder.
- Checkpoint SHA256 inventory is stored under `03_Audits_and_Checksums/sha256/`.
- Benchmark/cross-domain overlap notes are stored under `03_Audits_and_Checksums/overlap_checks/`.

REVIEW CONTENT BEFORE DECISION: `overlap_checks/SUBMISSION_STATUS_REVIEW_REQUIRED.md` reports SHA256 overlap that must be reconciled before the editor submission is marked ready.
