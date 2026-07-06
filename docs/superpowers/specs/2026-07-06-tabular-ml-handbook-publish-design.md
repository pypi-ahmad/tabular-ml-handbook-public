# Tabular ML Handbook Publish Design (2026-07-06)

**Owner:** Ahmad  
**Repo:** tabular-ml-handbook  
**Decision:** Publish to GitHub as a new **public** repository, with a verified end-to-end rerun.

## Goal

Ship a publish-ready version of this repository where:

- 7 of 8 notebooks are re-executed end-to-end on this machine, producing fresh outputs.
- The TabPFN v3 notebook is explicitly excluded from the E2E run (no token available).
- PDFs (notebook PDFs + docs PDFs) are rebuilt from the executed sources.
- `README.md` (and docs if needed) matches the executed notebook outputs and provides correct,
  reproducible run commands.
- The repository is pushed to a new public GitHub repo on the `main` branch.

## Non-Goals

- No new datasets or models.
- No refactor into a Python package; notebooks remain standalone.
- No TabPFN execution (requires manual license acceptance + `TABPFN_TOKEN`).

## Constraints / Environment Notes

- `uv` cannot write to `~/.cache/uv` in this environment; E2E runs must set `UV_CACHE_DIR` to a
  writable directory (e.g. `/tmp/uv-cache`) and similarly route common caches to `/tmp`.
- Use uv-managed Python `3.13.13` for the E2E run to match the notebooks/README claims.

## Success Criteria

- `jupyter nbconvert --execute` succeeds for these notebooks:
  - Google TabFM (classification + regression)
  - TabICLv2 (regression)
  - CatBoost (regression)
  - LightGBM (regression)
  - Mitra (classification)
  - XGBoost (classification)
- TabPFN notebook remains unexecuted and is documented as such.
- Rebuilt PDFs exist under `notebooks_pdf/` and `docs/*.pdf`.
- Clean git history with commits for spec, reruns, artifact rebuild, and README/doc sync.
- New public GitHub repo exists and `git push -u origin main` succeeds.

