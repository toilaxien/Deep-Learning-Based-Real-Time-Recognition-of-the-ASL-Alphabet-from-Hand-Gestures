# Dataset Documentation

This folder documents the benchmark and cross-domain datasets referenced by the paper.

## Public Benchmark Dataset

ASL Alphabet dataset:

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

The paper uses a 64:16:20 train/validation/test split.

## Independent Cross-Domain Dataset

The private cross-domain validation set contains:

- 870 RGB images.
- 29 ASL classes.
- 30 images per class.
- 5 signers.
- 2 consumer-grade cameras.
- 3 lighting conditions.
- Plain and cluttered backgrounds.

The full image set is not released publicly because it contains participant images collected under informed consent. Public representative samples are provided in `sample_cross_domain/`.

## Files

- `class_distribution.csv`: expected public/private class distribution.
- `cross_domain_metadata_template.csv`: metadata schema for the private 870-image set.
- `annotation_guidelines.md`: annotation protocol used for cross-domain labeling.
- `inter_annotator_agreement.md`: Cohen's kappa reporting note.
