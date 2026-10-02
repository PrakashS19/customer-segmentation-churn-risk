# GitHub browser upload edition

1. Create an empty public repository named `customer-segmentation-churn-risk`.
2. Extract this ZIP and use GitHub's `Add file -> Upload files` to upload the **contents** (not ZIP). Commit changes.
3. The ~45.6 MB raw Excel workbook is intentionally omitted because GitHub browser uploads support a maximum of 25 MiB per file. To run independently, download the UCI Online Retail II Excel source and save `online_retail_II.xlsx` inside `data/`. See `data/DOWNLOAD_DATA.md` for its verified SHA256.
4. The customer-level risk score CSV is also intentionally excluded from this publicly shared version; all aggregated charts and model results remain.
5. Open the GitHub README and check its embedded charts. Verify `notebooks/analysis_EXECUTED.ipynb`, `outputs/model_metrics.csv` and `images/model_performance.png` appear.

The other ZIP, `customer_segmentation_churn_GITHUB_VERIFIED.zip`, has **all files including the exact source Excel and customer-level output** and is best kept as your complete local reproducible archive. You can use Git CLI instead of GitHub browser upload if you want to publish the larger Excel.
