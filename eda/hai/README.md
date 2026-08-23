# HAI-23.05 EDA

This directory contains a reproducible, file-based EDA for the local
`HAI-23.05` data. It deliberately avoids paper-derived domain assumptions and
does not create the later 5–100% training subsets.

Run from the repository root with the available Python 3.8 installation:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python38\python.exe" eda\hai\run_hai_eda.py
```

Generated tables, figures, metadata, and the Korean summary report are written
under `eda/hai/outputs/`.
