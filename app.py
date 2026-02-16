<<<<<<< HEAD
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Multi-Sport Predictor", page_icon="🏆", layout="wide")

# Title
st.title("🏆 Multi-Sport Player Performance Predictor")
st.markdown("Predict player performance across Cricket, F1, NBA, NFL, and Soccer")

# ===== LOAD DATA FUNCTIONS =====
@st.cache_data
def load_cricket_data():
    matches = pd.read_csv('Cricket/matches_small.csv')
    deliveries = pd.read_csv('Cricket/deliveries_small.csv')
    
    batting_stats = deliveries.groupby(['match_id', 'batter']).agg({
        'batsman_runs': 'sum',
        'ball': 'count',
        'is_wicket': 'sum'
    }).reset_index()
    batting_stats.columns = ['match_id', 'player', 'runs', 'balls_faced', 'times_out']
    batting_stats['strike_rate'] = (batting_stats['runs'] / batting_stats['balls_faced']) * 100
    
    player_stats = batting_stats.groupby('player').agg({
        'runs': ['mean', 'std', 'max', 'count'],
        'balls_faced': 'mean',
        'strike_rate': 'mean'
    }).reset_index()
    player_stats.columns = ['player', 'avg_runs', 'std_runs', 'max_runs', 'matches_played', 'avg_balls', 'avg_strike_rate']
    player_stats = player_stats.fillna(0)
    player_stats = player_stats[player_stats['matches_played'] >= 10]
    
    return player_stats

@st.cache_data
def load_f1_data():
    drivers = pd.read_csv('F1/drivers.csv')
    results = pd.read_csv('F1/results.csv')
    races = pd.read_csv('F1/races.csv')
    
    f1_data = results.merge(races[['raceId', 'year', 'name']], on='raceId')
    f1_data = f1_data.merge(drivers[['driverId', 'forename', 'surname']], on='driverId')
    f1_data['driver_name'] = f1_data['forename'] + ' ' + f1_data['surname']
    f1_data['position'] = pd.to_numeric(f1_data['position'], errors='coerce')
    
    driver_stats = f1_data.groupby('driver_name').agg({
        'points': ['mean', 'sum', 'max', 'count'],
        'position': 'mean'
    }).reset_index()
    driver_stats.columns = ['player', 'avg_points', 'total_points', 'max_points', 'races', 'avg_position']
    driver_stats = driver_stats.fillna(0)
    driver_stats = driver_stats[driver_stats['races'] >= 10]
    
    return driver_stats

@st.cache_data
def load_nba_data():
    stats = pd.read_csv('NBA/Seasons_Stats_small.csv')
    stats = stats.dropna(subset=['Player', 'PTS', 'G'])
    stats = stats[stats['G'] > 0]
    
    player_stats = stats.groupby('Player').agg({
        'PTS': ['mean', 'sum', 'max'],
        'G': 'sum',
        'AST': 'mean',
        'TRB': 'mean'
    }).reset_index()
    player_stats.columns = ['player', 'avg_points', 'total_points', 'max_points', 'games', 'avg_assists', 'avg_rebounds']
    player_stats = player_stats.fillna(0)
    player_stats = player_stats[player_stats['games'] >= 20]
    
    return player_stats

@st.cache_data
def load_nfl_data():
    passing = pd.read_csv('NFL/Career_Stats_Passing.csv')
    
    passing['Games Played'] = pd.to_numeric(passing['Games Played'], errors='coerce')
    passing['Passing Yards'] = pd.to_numeric(passing['Passing Yards'], errors='coerce')
    passing['TD Passes'] = pd.to_numeric(passing['TD Passes'], errors='coerce')
    passing['Passer Rating'] = pd.to_numeric(passing['Passer Rating'], errors='coerce')
    
    passing = passing.dropna(subset=['Games Played', 'Passing Yards', 'TD Passes'])
    passing = passing[passing['Games Played'] > 0]
    
    player_stats = passing.groupby('Name').agg({
        'Passing Yards': ['mean', 'sum', 'max'],
        'TD Passes': ['mean', 'sum'],
        'Games Played': 'sum',
        'Passer Rating': 'mean'
    }).reset_index()
    player_stats.columns = ['player', 'avg_yards', 'total_yards', 'max_yards', 'avg_td', 'total_td', 'games', 'avg_rating']
    player_stats = player_stats.fillna(0)
    player_stats = player_stats[player_stats['games'] >= 10]
    
    return player_stats

@st.cache_data
def load_soccer_data():
    soccer_data = pd.read_csv('Soccer/soccer_small.csv')
    soccer_data = soccer_data.rename(columns={'player_name': 'player'})
    soccer_data = soccer_data.fillna(0)
    return soccer_data

# ===== TRAIN MODELS =====
@st.cache_resource
def train_cricket_model(player_stats):
    X = player_stats[['avg_balls', 'avg_strike_rate', 'matches_played', 'max_runs']]
    y = player_stats['avg_runs']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

@st.cache_resource
def train_f1_model(driver_stats):
    X = driver_stats[['races', 'avg_position', 'max_points', 'total_points']]
    y = driver_stats['avg_points']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

@st.cache_resource
def train_nba_model(player_stats):
    X = player_stats[['games', 'avg_assists', 'avg_rebounds', 'max_points']]
    y = player_stats['avg_points']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

@st.cache_resource
def train_nfl_model(player_stats):
    X = player_stats[['games', 'avg_td', 'avg_rating', 'max_yards']]
    y = player_stats['avg_yards']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

@st.cache_resource
def train_soccer_model(player_stats):
    X = player_stats[['finishing', 'dribbling', 'short_passing', 'shot_power', 
                      'sprint_speed', 'stamina', 'strength', 'vision']]
    y = player_stats['overall_rating']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# ===== SIDEBAR =====
st.sidebar.header("Select Sport")
sport = st.sidebar.radio("Choose a sport:", ["Cricket", "F1 Racing", "NBA Basketball", "NFL Football", "Soccer"])

# ===== MAIN CONTENT =====
if sport == "Cricket":
    st.header("🏏 Cricket Performance Predictor")
    
    with st.spinner("Loading Cricket data..."):
        cricket_stats = load_cricket_data()
        cricket_model = train_cricket_model(cricket_stats)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Player")
        player = st.selectbox("Choose a player:", sorted(cricket_stats['player'].unique()))
        
        if player:
            player_data = cricket_stats[cricket_stats['player'] == player].iloc[0]
            
            st.metric("Average Runs", f"{player_data['avg_runs']:.1f}")
            st.metric("Matches Played", f"{int(player_data['matches_played'])}")
            st.metric("Strike Rate", f"{player_data['avg_strike_rate']:.1f}")
            st.metric("Max Runs", f"{int(player_data['max_runs'])}")
    
    with col2:
        st.subheader("Prediction")
        if player:
            features = [[player_data['avg_balls'], player_data['avg_strike_rate'], 
                        player_data['matches_played'], player_data['max_runs']]]
            prediction = cricket_model.predict(features)[0]
            
            st.metric("Predicted Runs (Next Match)", f"{prediction:.1f}")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Predicted Runs"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 25], 'color': "lightgray"},
                           {'range': [25, 50], 'color': "gray"},
                           {'range': [50, 100], 'color': "lightgreen"}]}
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 10 Players by Average Runs")
    top_players = cricket_stats.nlargest(10, 'avg_runs')[['player', 'avg_runs', 'matches_played', 'avg_strike_rate']]
    st.dataframe(top_players, use_container_width=True)

elif sport == "F1 Racing":
    st.header("🏎️ F1 Performance Predictor")
    
    with st.spinner("Loading F1 data..."):
        f1_stats = load_f1_data()
        f1_model = train_f1_model(f1_stats)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Driver")
        driver = st.selectbox("Choose a driver:", sorted(f1_stats['player'].unique()))
        
        if driver:
            driver_data = f1_stats[f1_stats['player'] == driver].iloc[0]
            
            st.metric("Average Points", f"{driver_data['avg_points']:.1f}")
            st.metric("Total Races", f"{int(driver_data['races'])}")
            st.metric("Avg Position", f"{driver_data['avg_position']:.1f}")
            st.metric("Max Points", f"{int(driver_data['max_points'])}")
    
    with col2:
        st.subheader("Prediction")
        if driver:
            features = [[driver_data['races'], driver_data['avg_position'], 
                        driver_data['max_points'], driver_data['total_points']]]
            prediction = f1_model.predict(features)[0]
            
            st.metric("Predicted Points (Next Race)", f"{prediction:.1f}")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Predicted Points"},
                gauge={'axis': {'range': [0, 25]},
                       'bar': {'color': "red"},
                       'steps': [
                           {'range': [0, 8], 'color': "lightgray"},
                           {'range': [8, 15], 'color': "orange"},
                           {'range': [15, 25], 'color': "gold"}]}
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 10 Drivers by Average Points")
    top_drivers = f1_stats.nlargest(10, 'avg_points')[['player', 'avg_points', 'races', 'avg_position']]
    st.dataframe(top_drivers, use_container_width=True)

elif sport == "NBA Basketball":
    st.header("🏀 NBA Performance Predictor")
    
    with st.spinner("Loading NBA data..."):
        nba_stats = load_nba_data()
        nba_model = train_nba_model(nba_stats)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Player")
        player = st.selectbox("Choose a player:", sorted(nba_stats['player'].unique()))
        
        if player:
            player_data = nba_stats[nba_stats['player'] == player].iloc[0]
            
            st.metric("Average Points", f"{player_data['avg_points']:.1f}")
            st.metric("Total Games", f"{int(player_data['games'])}")
            st.metric("Avg Assists", f"{player_data['avg_assists']:.1f}")
            st.metric("Avg Rebounds", f"{player_data['avg_rebounds']:.1f}")
    
    with col2:
        st.subheader("Prediction")
        if player:
            features = [[player_data['games'], player_data['avg_assists'], 
                        player_data['avg_rebounds'], player_data['max_points']]]
            prediction = nba_model.predict(features)[0]
            
            st.metric("Predicted Points (Next Game)", f"{prediction:.1f}")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Predicted Points"},
                gauge={'axis': {'range': [0, 35]},
                       'bar': {'color': "orange"},
                       'steps': [
                           {'range': [0, 10], 'color': "lightgray"},
                           {'range': [10, 20], 'color': "lightblue"},
                           {'range': [20, 35], 'color': "gold"}]}
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 10 Players by Average Points")
    top_players = nba_stats.nlargest(10, 'avg_points')[['player', 'avg_points', 'games', 'avg_assists', 'avg_rebounds']]
    st.dataframe(top_players, use_container_width=True)

elif sport == "NFL Football":
    st.header("🏈 NFL Performance Predictor")
    
    with st.spinner("Loading NFL data..."):
        nfl_stats = load_nfl_data()
        nfl_model = train_nfl_model(nfl_stats)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Player")
        player = st.selectbox("Choose a player:", sorted(nfl_stats['player'].unique()))
        
        if player:
            player_data = nfl_stats[nfl_stats['player'] == player].iloc[0]
            
            st.metric("Avg Passing Yards", f"{player_data['avg_yards']:.1f}")
            st.metric("Total Games", f"{int(player_data['games'])}")
            st.metric("Avg TD Passes", f"{player_data['avg_td']:.1f}")
            st.metric("Passer Rating", f"{player_data['avg_rating']:.1f}")
    
    with col2:
        st.subheader("Prediction")
        if player:
            features = [[player_data['games'], player_data['avg_td'], 
                        player_data['avg_rating'], player_data['max_yards']]]
            prediction = nfl_model.predict(features)[0]
            
            st.metric("Predicted Yards (Next Game)", f"{prediction:.1f}")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Predicted Passing Yards"},
                gauge={'axis': {'range': [0, 400]},
                       'bar': {'color': "green"},
                       'steps': [
                           {'range': [0, 150], 'color': "lightgray"},
                           {'range': [150, 250], 'color': "lightgreen"},
                           {'range': [250, 400], 'color': "gold"}]}
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 10 Players by Average Passing Yards")
    top_players = nfl_stats.nlargest(10, 'avg_yards')[['player', 'avg_yards', 'games', 'avg_td', 'avg_rating']]
    st.dataframe(top_players, use_container_width=True)

elif sport == "Soccer":
    st.header("⚽ Soccer Performance Predictor")
    
    with st.spinner("Loading Soccer data..."):
        soccer_stats = load_soccer_data()
        soccer_model = train_soccer_model(soccer_stats)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Player")
        player = st.selectbox("Choose a player:", sorted(soccer_stats['player'].unique()))
        
        if player:
            player_data = soccer_stats[soccer_stats['player'] == player].iloc[0]
            
            st.metric("Overall Rating", f"{player_data['overall_rating']:.1f}")
            st.metric("Potential", f"{player_data['potential']:.1f}")
            st.metric("Finishing", f"{player_data['finishing']:.1f}")
            st.metric("Dribbling", f"{player_data['dribbling']:.1f}")
    
    with col2:
        st.subheader("Prediction")
        if player:
            features = [[player_data['finishing'], player_data['dribbling'], 
                        player_data['short_passing'], player_data['shot_power'],
                        player_data['sprint_speed'], player_data['stamina'],
                        player_data['strength'], player_data['vision']]]
            prediction = soccer_model.predict(features)[0]
            
            st.metric("Predicted Overall Rating", f"{prediction:.1f}")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Predicted Rating"},
                gauge={'axis': {'range': [40, 100]},
                       'bar': {'color': "purple"},
                       'steps': [
                           {'range': [40, 60], 'color': "lightgray"},
                           {'range': [60, 80], 'color': "lightblue"},
                           {'range': [80, 100], 'color': "gold"}]}
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 10 Players by Overall Rating")
    top_players = soccer_stats.nlargest(10, 'overall_rating')[['player', 'overall_rating', 'potential', 'finishing', 'dribbling']]
    st.dataframe(top_players, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Sports Covered:**")
st.sidebar.markdown("🏏 Cricket | 🏎️ F1 | 🏀 NBA | 🏈 NFL | ⚽ Soccer")
st.sidebar.markdown("---")
=======
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Multi-Sport Predictor", page_icon="🏆", layout="wide")

# Title
st.title("🏆 Multi-Sport Player Performance Predictor")
st.markdown("Predict player performance across Cricket, F1, NBA, NFL, and Soccer")

# ===== LOAD DATA FUNCTIONS =====
@st.cache_data
def load_cricket_data():
    matches = pd.read_csv('Cricket/matches_small.csv')
    deliveries = pd.read_csv('Cricket/deliveries_small.csv')
    
    batting_stats = deliveries.groupby(['match_id', 'batter']).agg({
        'batsman_runs': 'sum',
        'ball': 'count',
        'is_wicket': 'sum'
    }).reset_index()
    batting_stats.columns = ['match_id', 'player', 'runs', 'balls_faced', 'times_out']
    batting_stats['strike_rate'] = (batting_stats['runs'] / batting_stats['balls_faced']) * 100
    
    player_stats = batting_stats.groupby('player').agg({
        'runs': ['mean', 'std', 'max', 'count'],
        'balls_faced': 'mean',
        'strike_rate': 'mean'
    }).reset_index()
    player_stats.columns = ['player', 'avg_runs', 'std_runs', 'max_runs', 'matches_played', 'avg_balls', 'avg_strike_rate']
    player_stats = player_stats.fillna(0)
    player_stats = player_stats[player_stats['matches_played'] >= 10]
    
    return player_stats

@st.cache_data
def load_f1_data():
    drivers = pd.read_csv('F1/drivers.csv')
    results = pd.read_csv('F1/results.csv')
    races = pd.read_csv('F1/races.csv')
    
    f1_data = results.merge(races[['raceId', 'year', 'name']], on='raceId')
    f1_data = f1_data.merge(drivers[['driverId', 'forename', 'surname']], on='driverId')
    f1_data['driver_name'] = f1_data['forename'] + ' ' + f1_data['surname']
    f1_data['position'] = pd.to_numeric(f1_data['position'], errors='coerce')
    
    driver_stats = f1_data.groupby('driver_name').agg({
        'points': ['mean', 'sum', 'max', 'count'],
        'position': 'mean'
    }).reset_index()
    driver_stats.columns = ['player', 'avg_points', 'total_points', 'max_points', 'races', 'avg_position']
    driver_stats = driver_stats.fillna(0)
    driver_stats = driver_stats[driver_stats['races'] >= 10]
    
    return driver_stats

@st.cache_data
def load_nba_data():
    stats = pd.read_csv('NBA/Seasons_Stats_small.csv')
    stats = stats.dropna(subset=['Player', 'PTS', 'G'])
    stats = stats[stats['G'] > 0]
    
    player_stats = stats.groupby('Player').agg({
        'PTS': ['mean', 'sum', 'max'],
        'G': 'sum',
        'AST': 'mean',
        'TRB': 'mean'
    }).reset_index()
    player_stats.columns = ['player', 'avg_points', 'total_points', 'max_points', 'games', 'avg_assists', 'avg_rebounds']
    player_stats = player_stats.fillna(0)
    player_stats = player_stats[player_stats['games'] >= 20]
    
    return player_stats

@st.cache_data
def load_nfl_data():
    passing = pd.read_csv('NFL/Career_Stats_Passing.csv')
    
    passing['Games Played'] = pd.to_numeric(passing['Games Played'], errors='coerce')
    passing['Passing Yards'] = pd.to_numeric(passing['Passing Yards'], errors='coerce')
    passing['TD Passes'] = pd.to_numeric(passing['TD Passes'], errors='coerce')
    passing['Passer Rating'] = pd.to_numeric(passing['Passer Rating'], errors='coerce')
    
    passing = passing.dropna(subset=['Games Played', 'Passing Yards', 'TD Passes'])
    passing = passing[passing['Games Played'] > 0]
    
    player_stats = passing.groupby('Name').agg({
        'Passing Yards': ['mean', 'sum', 'max'],
        'TD Passes': ['mean', 'sum'],
        'Games Played': 'sum',
        'Passer Rating': 'mean'
    }).reset_index()
    player_stats.columns = ['player', 'avg_yards', 'total_yards', 'max_yards', 'avg_td', 'total_td', 'games', 'avg_rating']
    player_stats = player_stats.fillna(0)
    player_stats = player_stats[player_stats['games'] >= 10]
    
    return player_stats

@st.cache_data
def load_soccer_data():
    soccer_data = pd.read_csv('Soccer/soccer_small.csv')
    soccer_data = soccer_data.rename(columns={'player_name': 'player'})
    soccer_data = soccer_data.fillna(0)
    return soccer_data

# ===== TRAIN MODELS =====
@st.cache_resource
def train_cricket_model(player_stats):
    X = player_stats[['avg_balls', 'avg_strike_rate', 'matches_played', 'max_runs']]
    y = player_stats['avg_runs']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

@st.cache_resource
def train_f1_model(driver_stats):
    X = driver_stats[['races', 'avg_position', 'max_points', 'total_points']]
    y = driver_stats['avg_points']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

@st.cache_resource
def train_nba_model(player_stats):
    X = player_stats[['games', 'avg_assists', 'avg_rebounds', 'max_points']]
    y = player_stats['avg_points']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

@st.cache_resource
def train_nfl_model(player_stats):
    X = player_stats[['games', 'avg_td', 'avg_rating', 'max_yards']]
    y = player_stats['avg_yards']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

@st.cache_resource
def train_soccer_model(player_stats):
    X = player_stats[['finishing', 'dribbling', 'short_passing', 'shot_power', 
                      'sprint_speed', 'stamina', 'strength', 'vision']]
    y = player_stats['overall_rating']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# ===== SIDEBAR =====
st.sidebar.header("Select Sport")
sport = st.sidebar.radio("Choose a sport:", ["Cricket", "F1 Racing", "NBA Basketball", "NFL Football", "Soccer"])

# ===== MAIN CONTENT =====
if sport == "Cricket":
    st.header("🏏 Cricket Performance Predictor")
    
    with st.spinner("Loading Cricket data..."):
        cricket_stats = load_cricket_data()
        cricket_model = train_cricket_model(cricket_stats)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Player")
        player = st.selectbox("Choose a player:", sorted(cricket_stats['player'].unique()))
        
        if player:
            player_data = cricket_stats[cricket_stats['player'] == player].iloc[0]
            
            st.metric("Average Runs", f"{player_data['avg_runs']:.1f}")
            st.metric("Matches Played", f"{int(player_data['matches_played'])}")
            st.metric("Strike Rate", f"{player_data['avg_strike_rate']:.1f}")
            st.metric("Max Runs", f"{int(player_data['max_runs'])}")
    
    with col2:
        st.subheader("Prediction")
        if player:
            features = [[player_data['avg_balls'], player_data['avg_strike_rate'], 
                        player_data['matches_played'], player_data['max_runs']]]
            prediction = cricket_model.predict(features)[0]
            
            st.metric("Predicted Runs (Next Match)", f"{prediction:.1f}")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Predicted Runs"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 25], 'color': "lightgray"},
                           {'range': [25, 50], 'color': "gray"},
                           {'range': [50, 100], 'color': "lightgreen"}]}
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 10 Players by Average Runs")
    top_players = cricket_stats.nlargest(10, 'avg_runs')[['player', 'avg_runs', 'matches_played', 'avg_strike_rate']]
    st.dataframe(top_players, use_container_width=True)

elif sport == "F1 Racing":
    st.header("🏎️ F1 Performance Predictor")
    
    with st.spinner("Loading F1 data..."):
        f1_stats = load_f1_data()
        f1_model = train_f1_model(f1_stats)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Driver")
        driver = st.selectbox("Choose a driver:", sorted(f1_stats['player'].unique()))
        
        if driver:
            driver_data = f1_stats[f1_stats['player'] == driver].iloc[0]
            
            st.metric("Average Points", f"{driver_data['avg_points']:.1f}")
            st.metric("Total Races", f"{int(driver_data['races'])}")
            st.metric("Avg Position", f"{driver_data['avg_position']:.1f}")
            st.metric("Max Points", f"{int(driver_data['max_points'])}")
    
    with col2:
        st.subheader("Prediction")
        if driver:
            features = [[driver_data['races'], driver_data['avg_position'], 
                        driver_data['max_points'], driver_data['total_points']]]
            prediction = f1_model.predict(features)[0]
            
            st.metric("Predicted Points (Next Race)", f"{prediction:.1f}")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Predicted Points"},
                gauge={'axis': {'range': [0, 25]},
                       'bar': {'color': "red"},
                       'steps': [
                           {'range': [0, 8], 'color': "lightgray"},
                           {'range': [8, 15], 'color': "orange"},
                           {'range': [15, 25], 'color': "gold"}]}
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 10 Drivers by Average Points")
    top_drivers = f1_stats.nlargest(10, 'avg_points')[['player', 'avg_points', 'races', 'avg_position']]
    st.dataframe(top_drivers, use_container_width=True)

elif sport == "NBA Basketball":
    st.header("🏀 NBA Performance Predictor")
    
    with st.spinner("Loading NBA data..."):
        nba_stats = load_nba_data()
        nba_model = train_nba_model(nba_stats)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Player")
        player = st.selectbox("Choose a player:", sorted(nba_stats['player'].unique()))
        
        if player:
            player_data = nba_stats[nba_stats['player'] == player].iloc[0]
            
            st.metric("Average Points", f"{player_data['avg_points']:.1f}")
            st.metric("Total Games", f"{int(player_data['games'])}")
            st.metric("Avg Assists", f"{player_data['avg_assists']:.1f}")
            st.metric("Avg Rebounds", f"{player_data['avg_rebounds']:.1f}")
    
    with col2:
        st.subheader("Prediction")
        if player:
            features = [[player_data['games'], player_data['avg_assists'], 
                        player_data['avg_rebounds'], player_data['max_points']]]
            prediction = nba_model.predict(features)[0]
            
            st.metric("Predicted Points (Next Game)", f"{prediction:.1f}")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Predicted Points"},
                gauge={'axis': {'range': [0, 35]},
                       'bar': {'color': "orange"},
                       'steps': [
                           {'range': [0, 10], 'color': "lightgray"},
                           {'range': [10, 20], 'color': "lightblue"},
                           {'range': [20, 35], 'color': "gold"}]}
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 10 Players by Average Points")
    top_players = nba_stats.nlargest(10, 'avg_points')[['player', 'avg_points', 'games', 'avg_assists', 'avg_rebounds']]
    st.dataframe(top_players, use_container_width=True)

elif sport == "NFL Football":
    st.header("🏈 NFL Performance Predictor")
    
    with st.spinner("Loading NFL data..."):
        nfl_stats = load_nfl_data()
        nfl_model = train_nfl_model(nfl_stats)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Player")
        player = st.selectbox("Choose a player:", sorted(nfl_stats['player'].unique()))
        
        if player:
            player_data = nfl_stats[nfl_stats['player'] == player].iloc[0]
            
            st.metric("Avg Passing Yards", f"{player_data['avg_yards']:.1f}")
            st.metric("Total Games", f"{int(player_data['games'])}")
            st.metric("Avg TD Passes", f"{player_data['avg_td']:.1f}")
            st.metric("Passer Rating", f"{player_data['avg_rating']:.1f}")
    
    with col2:
        st.subheader("Prediction")
        if player:
            features = [[player_data['games'], player_data['avg_td'], 
                        player_data['avg_rating'], player_data['max_yards']]]
            prediction = nfl_model.predict(features)[0]
            
            st.metric("Predicted Yards (Next Game)", f"{prediction:.1f}")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Predicted Passing Yards"},
                gauge={'axis': {'range': [0, 400]},
                       'bar': {'color': "green"},
                       'steps': [
                           {'range': [0, 150], 'color': "lightgray"},
                           {'range': [150, 250], 'color': "lightgreen"},
                           {'range': [250, 400], 'color': "gold"}]}
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 10 Players by Average Passing Yards")
    top_players = nfl_stats.nlargest(10, 'avg_yards')[['player', 'avg_yards', 'games', 'avg_td', 'avg_rating']]
    st.dataframe(top_players, use_container_width=True)

elif sport == "Soccer":
    st.header("⚽ Soccer Performance Predictor")
    
    with st.spinner("Loading Soccer data..."):
        soccer_stats = load_soccer_data()
        soccer_model = train_soccer_model(soccer_stats)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Player")
        player = st.selectbox("Choose a player:", sorted(soccer_stats['player'].unique()))
        
        if player:
            player_data = soccer_stats[soccer_stats['player'] == player].iloc[0]
            
            st.metric("Overall Rating", f"{player_data['overall_rating']:.1f}")
            st.metric("Potential", f"{player_data['potential']:.1f}")
            st.metric("Finishing", f"{player_data['finishing']:.1f}")
            st.metric("Dribbling", f"{player_data['dribbling']:.1f}")
    
    with col2:
        st.subheader("Prediction")
        if player:
            features = [[player_data['finishing'], player_data['dribbling'], 
                        player_data['short_passing'], player_data['shot_power'],
                        player_data['sprint_speed'], player_data['stamina'],
                        player_data['strength'], player_data['vision']]]
            prediction = soccer_model.predict(features)[0]
            
            st.metric("Predicted Overall Rating", f"{prediction:.1f}")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Predicted Rating"},
                gauge={'axis': {'range': [40, 100]},
                       'bar': {'color': "purple"},
                       'steps': [
                           {'range': [40, 60], 'color': "lightgray"},
                           {'range': [60, 80], 'color': "lightblue"},
                           {'range': [80, 100], 'color': "gold"}]}
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 10 Players by Overall Rating")
    top_players = soccer_stats.nlargest(10, 'overall_rating')[['player', 'overall_rating', 'potential', 'finishing', 'dribbling']]
    st.dataframe(top_players, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Sports Covered:**")
st.sidebar.markdown("🏏 Cricket | 🏎️ F1 | 🏀 NBA | 🏈 NFL | ⚽ Soccer")
st.sidebar.markdown("---")
>>>>>>> 7ed44a78d5fcaec78dc37f27eacbcbbb51828260
st.sidebar.markdown("*For educational purposes only*")