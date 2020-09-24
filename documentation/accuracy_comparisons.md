These accuracy measures were taken by batch training the en_core_web_sm model to add the INSTRUMENT and SPACECRAFT entity types.
Comparisons are made by training the model with both entity types and then each individual entity type without saving the model.

| Number of accept/reject examples | INSTRUMENT | SPACECRAFT | Both  |
|----------------------------------|------------|------------|-------|
| 327                              | 0.657      | 0.594      | 0.636 |
| 393                              | 0.776      | 0.765      | 0.773 |
