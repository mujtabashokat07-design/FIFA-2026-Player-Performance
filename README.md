# FIFA 2026 Player Performance AI Dashboard

A production-ready, glassmorphism-styled Streamlit app for analytics + ML on the
FIFA World Cup 2026 player performance dataset (54,600 rows × 75 columns).

## Run locally / Google Colab

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place `fifa_world_cup_2026_player_performance.csv` in the same folder as `app.py`
(already included).

In Google Colab, use `pyngrok` or `localtunnel` to expose the Streamlit server,
since Colab doesn't support `streamlit run` directly in the notebook cell.

## Deploy to Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and the CSV to a GitHub repo.
2. Go to https://share.streamlit.io, connect the repo, set main file to `app.py`.
3. Deploy — no extra config needed.

## App sections

- 🏠 **Home Dashboard** – KPIs, nation/position breakdowns
- 📊 **Data Exploration** – filterable/searchable data table + column stats
- 📈 **EDA** – top scorers, country comparisons, age vs performance, correlation
  heatmap, distributions, position violin plots
- 🤖 **Machine Learning** – trains Linear Regression, Random Forest, and Decision
  Tree regressors to predict `performance_score` or `goals`, with model comparison
- 🔮 **Prediction** – live form to forecast a player's performance
- 📊 **Model Insights** – feature importance + plain-language model explanations

## Notes on leakage-safe modeling

Composite/aggregate columns that are near-duplicates of the target
(`player_rating`, `tournament_rating`, `offensive_contribution`, etc.) are
excluded from the feature set so the models learn from raw match stats instead
of other derived scores.
