# Cross-Domain Annotation Guidelines

The independent cross-domain set is annotated at the image level using the 29 ASL Alphabet classes:

`A-Z`, `del`, `nothing`, and `space`.

## Protocol

1. Each image is assigned exactly one class label.
2. Two annotators independently label every image.
3. Disagreements are reviewed by a third annotator.
4. The final label is the resolved consensus label.
5. Images with severe blur, occlusion, or ambiguous hand configuration should be flagged in the `notes` column.

## Ambiguous Classes

Special attention should be given to morphologically similar classes:

`A`, `E`, `M`, `N`, `S`, and `T`.

These classes differ by subtle thumb placement and finger spacing, and they form the main error cluster discussed in the paper.

## Metadata Fields

Use `cross_domain_metadata_template.csv` as the metadata schema. The full private dataset should contain one metadata row per image.
