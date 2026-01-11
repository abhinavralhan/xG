# xG
An attempt at finding expected goals in a soccer game

# Expected Goals (xG) Model

Statistical model to quantify shot quality in football. Calculates the probability of a shot resulting in a goal based on location, angle, and match context.

## What is Expected Goals?

xG assigns a probability (0 to 1) to every shot attempt based on historical data. A penalty kick typically has an xG of ~0.76, while a header from outside the box might be 0.02.

**Example:** If a player takes 10 shots with xG values summing to 2.65 and scores 4 goals, they overperformed their expected output.

## Key Factors

| Factor | Impact | Why It Matters |
|--------|--------|----------------|
| Distance to Goal | High | Shots from closer range have significantly higher conversion rates |
| Angle to Goal | High | Central positions offer better target visibility than tight angles |
| Body Part | Medium | Foot shots convert better than headers on average |
| Shot Type | Medium | Open play vs. set piece vs. penalty affects probability |

## Model Approach

This implementation uses **Logistic Regression** trained on StatsBomb open data:

1. **Feature Engineering**: Distance, angle, body part, shot type, previous action
2. **Training**: Logistic regression with class balancing (goals are rare events)
3. **Evaluation**: Log-loss, calibration plots, ROC-AUC
4. **Visualization**: Shot maps with xG values overlaid

## Usage

```python
# Load the trained model
import pickle
with open('xg_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict xG for a shot
features = {
    'distance': 12.5,  # meters from goal
    'angle': 0.45,     # radians
    'body_part': 'right_foot',
    'shot_type': 'open_play'
}
xg = model.predict_proba(features)[0][1]
```

## Data Source

Uses [StatsBomb Open Data](https://github.com/statsbomb/open-data) - free football event data including shot locations, outcomes, and context.

## Tech Stack

- Python
- Jupyter Notebook
- Scikit-learn (Logistic Regression)
- Pandas / NumPy
- Matplotlib / mplsoccer (visualization)

## Results

The model achieves reasonable calibration on held-out test data, with shots predicted at 0.1 xG converting at approximately 10%.

## References

- [StatsBomb xG Methodology](https://statsbomb.com/soccer-metrics/expected-goals-xg-explained/)
- [Understat](https://understat.com/) - Public xG data
- Original xG research by Opta and others

## License

MIT
