# Inter-Annotator Agreement

The paper reports Cohen's kappa:

```text
kappa = 0.94
```

This value indicates high agreement between the two independent annotators before third-annotator resolution.

To reproduce this value, use the private cross-domain metadata file with the columns:

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

The public repository includes only representative samples, so the exact kappa calculation requires the private 870-image metadata file.
