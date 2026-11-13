# AI Scouting Platform ⚽

An interactive **Streamlit-based football scouting application** for exploring player statistics, comparing players, and generating data-driven scouting summaries.

## Features

- Upload player statistics in **CSV, XLSX, or XLS** format
- Run the app with **reproducible fictional sample data** without uploading a dataset
- Analyze players by season, club, and position group
- Calculate season- and position-group-specific standardized metrics (Z-scores)
- Adjust metric weights and examine relative scouting scores and grades
- Inspect player strengths, weaknesses, and rule-based playing-style labels
- Visualize player profiles with Plotly radar charts and multi-season trends
- Compare multiple players and identify similar players using K-Means clusters and cosine similarity

## Getting started

Requires Python 3.10+ (recommended).

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

When the app opens, leave **가상 샘플 데이터로 실행** checked to explore the interface with fictional records. You can upload your own compatible data instead.

## Project structure

```text
AI-Scouting-Platform/
├── app.py                 # Streamlit interface and scoring pipeline
├── analysis_utils.py      # Standardization, history and recommendation helpers
├── sample_data.py         # Fictional demonstration dataset generator
├── requirements.txt       # Python package dependencies
├── .gitignore
└── README.md
```

## Methodology and interpretation

- Scores are **relative to the available season and position-group comparison set**, not absolute measures of ability.
- User-configurable weights affect rankings and grades; the resulting scores are not constrained to 0–100.
- Playing-style and detailed-position labels are **rule-based estimates**, not validated predictions.
- Player recommendations use standardized statistical features, K-Means grouping and cosine similarity. Similarity is **not** a probability of recruitment success.
- Historical comparisons are relative to each season's reference group. Matching solely by player name may combine namesakes.
- Minimum playing-time filters and dataset coverage can change the comparison population and results.

## Data and reproducibility

The included `sample_data.py` creates fictional player records with a fixed random seed. No licensed or private player dataset is included. Users are responsible for ensuring they have rights to any real datasets they upload or redistribute.

Dependencies were derived from the imported packages and supported Excel readers. Exact package versions and end-to-end deployment behavior have not yet been verified.

## Author

**Junha Shin** — [GitHub](https://github.com/sinjunha513-cloud)
