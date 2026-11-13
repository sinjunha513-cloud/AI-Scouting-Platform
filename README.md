# AI Scouting Platform ⚽

An interactive **Streamlit-based football scouting application** for exploring player statistics, comparing players, and generating data-driven scouting summaries.

## Live demo

**[Launch the AI Scouting Platform](https://junha-ai-scouting.streamlit.app/)**

Try the app with the built-in fictional sample dataset—no real player data is required.

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
├── test_analysis.py        # Model and Streamlit regression tests
├── .gitignore
└── README.md
```

## Methodology and interpretation

- Scores are **relative to the available season and position-group comparison set**, not absolute measures of ability.
- User-configurable weights affect rankings and grades; the resulting scores are not constrained to 0–100.
- Playing-style and detailed-position labels are **rule-based estimates**, not validated predictions. Their standardized inputs are independent of user scoring weights. Style percentages express relative rule scores, not calibrated probabilities.
- Player recommendations use standardized statistical features, K-Means grouping and cosine similarity. Similarity is **not** a probability of recruitment success.
- Historical comparisons are relative to each season's reference group. Matching solely by player name may combine namesakes.
- Minimum playing-time filters and dataset coverage can change the comparison population and results.
- Missing per-90 counts are derived from cumulative counts and positive playing time when possible. Cumulative counts are never used as a fallback in per-90 scoring; success percentages are not divided by minutes.
- Missing observations remain missing in source metrics. Scoring uses a neutral Z-score of zero for missing inputs; strengths and weaknesses use observed values only. The interface shows the observed share of scoring weights.
- Players without any observed, positively weighted scoring metric are excluded. If no evaluable data remains, the app displays guidance instead of assigning default scores.
- Strengths and weaknesses are ordered by deviation in the favorable or unfavorable direction, including lower-is-better metrics such as fouls and ball losses.

## Data and reproducibility

The included `sample_data.py` creates fictional player records with a fixed random seed. No licensed or private player dataset is included. Users are responsible for ensuring they have rights to any real datasets they upload or redistribute.

Install the dependencies in `requirements.txt`, then run the regression suite:

```bash
python -m unittest test_analysis.py
```

The suite covers per-90 derivation, missing observations, scoring-independent role labels, lower-is-better strengths and weaknesses, player recommendations, and Streamlit screens with missing metrics or all-zero weights. An additional local-workbook check runs when an XLSX file is available and is skipped otherwise. These checks validate implementation behavior, not real-world scouting accuracy.

## Author

**Junha Shin** — [GitHub](https://github.com/sinjunha513-cloud)
