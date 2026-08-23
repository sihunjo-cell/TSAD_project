# GHL EDA

This directory contains a reproducible exploratory analysis for the 25 local
GHL multivariate time-series files stored in `TSB-AD-M`.

The GHL files do not contain a timestamp column. The analysis therefore uses
the zero-based row index as the time axis and treats every file as a separate
time-series boundary. Files are never connected for lag or window operations,
but they are not assumed to be statistically independent because they share
the same GHL process-model provenance. Filename metadata (`tr_<n>` and
`1st_<n>`) is parsed and validated against the embedded `Label` column.

Run from the repository root with the available Python 3.8 installation:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python38\python.exe" eda\ghl\run_ghl_eda.py
```

Generated tables, figures, metadata, and Korean reports are written under
`eda/ghl/outputs/`.

`GHL_EDA_NOTION_REPORT.md` is the edited Notion handoff with explicit visual
insertion markers. Reruns write the reproducible automatic draft to
`GHL_EDA_NOTION_REPORT_AUTO.md`, preserving the edited handoff.
