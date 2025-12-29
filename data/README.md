# data/

This folder holds **versioned inputs** used by the MLBC projection pipeline.

Recommended workflow for Baseball Savant park factor priors:

1. Fetch/update a Savant prior CSV (checked into git for reproducibility)
2. Load it into the SQLite DB as a prior table
3. Compute MLBC park factors, blending MLBC observed with Savant priors

See `savant_park_factors_to_csv.py` and `mlbc_parkfactors.py`.
