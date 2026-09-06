# Inter-Annotator Agreement

Inter-annotator agreement should be computed from the private canonical 870-image cross-domain metadata.

The public repository does not include the private per-image annotator labels, so this file does not assert a specific Cohen's kappa value. If a kappa value is reported in the manuscript, include the private calculation evidence in the editor package or mark the value as requiring update.

Expected private metadata columns:

- `annotator_1_label`
- `annotator_2_label`

Example calculation:

```python
import pandas as pd
from sklearn.metrics import cohen_kappa_score

df = pd.read_csv("cross_domain_metadata_private.csv")
kappa = cohen_kappa_score(df["annotator_1_label"], df["annotator_2_label"])
print(kappa)
```

REVIEW CONTENT BEFORE DECISION: add the verified kappa value only after confirming it was computed on the current canonical 870-image cross-domain set.
