# Data

```text
x, y, pos, age, netuse, treated, green, phc
```

Column meanings:

- `x`, `y`: village spatial coordinates
- `pos`: binary malaria outcome, where 1 indicates malaria-positive
- `age`: child age
- `netuse`: bed net use indicator
- `treated`: treatment indicator
- `green`: vegetation/environmental measure
- `phc`: primary health center access indicator

The original R example used the `gambia` dataset from the `geoR` package.

The geographic boundary data (`gambia_borders.csv`) was extracted from the original R `geoR` dataset and converted to CSV format for use in Python-based visualizations. This allows replication of the original spatial plots without requiring R dependencies.