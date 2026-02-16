# 🏆 Multi-Sport Player Performance Predictor

A machine learning web application that predicts player performance across 5 major sports: Cricket, Formula 1, NBA, NFL, and Soccer.

## 🚀 Live Demo

**[Click here to try the app](YOUR_STREAMLIT_URL_HERE)**

## ✨ Features

### 🏏 Cricket
- Predict runs for next match
- View batting stats (average, strike rate, max runs)
- Top 10 players leaderboard

### 🏎️ Formula 1
- Predict points for next race
- View driver stats (avg points, races, position)
- Top 10 drivers leaderboard

### 🏀 NBA Basketball
- Predict points for next game
- View player stats (points, assists, rebounds)
- Top 10 players leaderboard

### 🏈 NFL Football
- Predict passing yards for next game
- View quarterback stats (yards, TD passes, passer rating)
- Top 10 players leaderboard

### ⚽ Soccer
- Predict overall player rating
- View player attributes (finishing, dribbling, passing)
- Top 10 players leaderboard

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend language |
| Streamlit | Web application framework |
| Scikit-learn | Machine learning (Random Forest) |
| Pandas | Data manipulation |
| Plotly | Interactive visualizations |

## 📊 Datasets

| Sport | Source | Size |
|-------|--------|------|
| Cricket | IPL Dataset (Kaggle) | 100,000+ deliveries |
| F1 | Formula 1 Dataset (Kaggle) | 26,000+ race results |
| NBA | NBA Players Stats (Kaggle) | 8,000+ season stats |
| NFL | NFL Statistics (Kaggle) | 1,000+ player stats |
| Soccer | European Soccer (Kaggle) | 11,000+ players |

## 🔍 How It Works

1. **Data Loading**: Loads historical player/driver performance data
2. **Feature Engineering**: Calculates averages, totals, and performance metrics
3. **Model Training**: Trains Random Forest Regressor for each sport
4. **Prediction**: Predicts next game/match/race performance
5. **Visualization**: Displays predictions with interactive gauge charts

## 🏃 Run Locally

```bash
# Clone the repository
git clone https://github.com/yashsingh05/multi-sport-predictor.git
cd multi-sport-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📁 Project Structure

```
multi-sport-predictor/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── Cricket/
│   ├── matches_small.csv       # Cricket matches data
│   └── deliveries_small.csv    # Ball-by-ball data
├── F1/
│   ├── drivers.csv             # F1 drivers data
│   ├── results.csv             # Race results
│   └── races.csv               # Race information
├── NBA/
│   ├── Players.csv             # NBA players info
│   └── Seasons_Stats_small.csv # Season statistics
├── NFL/
│   └── Career_Stats_Passing.csv # Quarterback stats
└── Soccer/
    └── soccer_small.csv        # Player attributes
```

## 📈 Model Performance

| Sport | Model | Target Prediction |
|-------|-------|-------------------|
| Cricket | Random Forest | Runs per match |
| F1 | Random Forest | Points per race |
| NBA | Random Forest | Points per game |
| NFL | Random Forest | Passing yards per game |
| Soccer | Random Forest | Overall rating |

## 🔮 Future Enhancements

- [ ] Add more sports (Tennis, Golf, MLB)
- [ ] Include real-time data updates
- [ ] Add player comparison feature
- [ ] Implement fantasy sports scoring
- [ ] Add injury and form factors
- [ ] Deploy mobile app version

## ⚠️ Disclaimer

This application is for **educational and entertainment purposes only**. Predictions are based on historical data and should not be used for betting or financial decisions.

## 👤 Author

**Yash Singh**

- GitHub: [@yashsingh05](https://github.com/yashsingh05)



---

⭐ If you found this project useful, please give it a star!
