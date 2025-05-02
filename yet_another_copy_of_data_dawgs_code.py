!pip install nba_api

from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2, boxscoresummaryv2
import pandas as pd
import time

# First, pull all games from the 2023–24 season
from nba_api.stats.endpoints import leaguegamefinder

gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable='2023-24',league_id_nullable='00')
games_df = gamefinder.get_data_frames()[0]

# Filter to the Dallas vs. Houston game on April 7, 2024
selected = games_df[(games_df['GAME_DATE'] == '2024-04-07') & (games_df['MATCHUP'] == 'DAL vs. HOU')]

# Grab its GAME_ID
game_id = selected.iloc[0]['GAME_ID']

# Display TEAM_NAME, PTS and WL instead of HOME_TEAM_NAME/VISITOR_TEAM_NAME for readability
print("Selected game details:")
print(selected[['GAME_DATE', 'MATCHUP', 'TEAM_NAME', 'PTS', 'WL']])

# Get box score and summary data for the selected game
boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)

# Player-level stats
player_stats = boxscore.get_data_frames()[0]

# Correct team-level quarter scores (LINE_SCORE is at index 5)
line_score = summary.get_data_frames()[5]

# If PTS_OT1 doesn't exist (rare), set it to 0
if 'PTS_OT1' not in line_score.columns:
    line_score['PTS_OT1'] = 0

# Define team1/team2 from line_score
team1 = line_score.iloc[0]
team2 = line_score.iloc[1]

# Display relevant info
line_score[['TEAM_ABBREVIATION', 'PTS_QTR1', 'PTS_QTR2', 'PTS_QTR3', 'PTS_QTR4', 'PTS_OT1', 'PTS']]

from nba_api.stats.endpoints import playbyplayv2
import pandas as pd
import numpy as np

# The play-by-play preprocess
pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
pbp_df = pbp.get_data_frames()[0]
pbp_df['SCOREMARGIN'] = pd.to_numeric(pbp_df['SCOREMARGIN'], errors='coerce')
pbp_df['SECONDS_LEFT'] = (pbp_df['PCTIMESTRING'].str.split(':').str[0].astype(float) * 60 + pbp_df['PCTIMESTRING'].str.split(':').str[1].astype(float))

# Total-Game Close Score Bonus (tiered on final margin)
final_diff = abs(team1['PTS'] - team2['PTS'])
if final_diff <= 5:
    diff_score = 95
elif final_diff <= 10:
    diff_score = 85
elif final_diff <= 20:
    diff_score = 30
else:
    diff_score = 0

# Star Performance (30+ points gives 75 points each)
num_star_30 = len(player_stats[player_stats['PTS'] >= 30])
star_score = 75 * num_star_30

# Big Comeback (down ≥20 at half & win gives 88 points)
half1_t1 = team1['PTS_QTR1'] + team1['PTS_QTR2']
half1_t2 = team2['PTS_QTR1'] + team2['PTS_QTR2']
down_by = abs(half1_t1 - half1_t2)
leader_at_half = 'team1' if half1_t1 > half1_t2 else 'team2'
winner = 'team1' if team1['PTS'] > team2['PTS'] else 'team2'
was_big_comeback = (down_by >= 20 and leader_at_half != winner)
comeback_score = 88 if was_big_comeback else 0

# Overtime Bonus (any OT points gives 98 points)
ot_score = 98 if (team1.get('PTS_OT1', 0) > 0 or team2.get('PTS_OT1', 0) > 0) else 0

# True Lead-Flips Calculation
pbp_df['SCOREMARGIN_NUM'] = pbp_df['SCOREMARGIN']
pbp_df['SIGN'] = pbp_df['SCOREMARGIN_NUM'].apply(lambda x: 1 if x>0 else (-1 if x<0 else np.nan))
pbp_df['PREV_SIGN'] = pbp_df['SIGN'].shift()
pbp_df['FLIP'] = (pbp_df['SIGN'] * pbp_df['PREV_SIGN'] == -1)
pbp_df['HALF'] = pbp_df['PERIOD'].apply(lambda p: 1 if p<=2 else 2)

first_half_flips = int(pbp_df[(pbp_df['HALF']==1) & pbp_df['FLIP']].shape[0])
second_half_flips = int(pbp_df[(pbp_df['HALF']==2) & pbp_df['FLIP']].shape[0])
first_half_score = 15 * first_half_flips
second_half_score = 25 * second_half_flips

# Final-Minute FGAs (last 5 min Q4, margin <10)
fga_plays = pbp_df[(pbp_df['PERIOD']==4) & (pbp_df['SECONDS_LEFT']<=300) & (pbp_df['EVENTMSGTYPE']==1) & (pbp_df['SCOREMARGIN_NUM'].abs()<10)]

attempts = len(fga_plays)
made = 0
for idx, row in fga_plays.iterrows():
    old = row['SCOREMARGIN_NUM']
    nxt = pbp_df.loc[idx+1:, 'SCOREMARGIN_NUM'].dropna().iloc[0]
    if abs(nxt) != abs(old):
        made += 1
attempts_score = 15 * attempts
makes_score    = 30 * made

# Pressure Makes: every made FG or FT (EVENTMSGTYPE 1 or 3) with margin<10 ⇒ 10 points each
pressure_makes = pbp_df[(pbp_df['EVENTMSGTYPE'].isin([1,3])) & (pbp_df['SCOREMARGIN_NUM'].abs() < 10)].shape[0]
pressure_score = 10 * pressure_makes

# Final Entertainment Index
entertainment_score = (diff_score + star_score + comeback_score + ot_score + first_half_score + second_half_score + attempts_score + makes_score + pressure_score)

# Print Detailed Breakdown
print("Final Metric Breakdown (0–100+ scale):")
print(f"Total-Game Close Bonus: {diff_score} (margin={final_diff})")
print(f"Star 30+ (×75 each): {star_score}  ({num_star_30} player(s))")
print(f"Big Comeback (≥20↓): {comeback_score}")
print(f"Overtime Bonus: {ot_score}")
print(f"1H Lead-Flips (×15): {first_half_flips} → {first_half_score}")
print(f"2H Lead-Flips (×25): {second_half_flips} → {second_half_score}")
print(f"FGA Attempts (×15): {attempts} → {attempts_score}")
print(f"Makes in those FGAs (×30): {made} → {makes_score}")
print(f"Pressure Makes (×10): {pressure_makes} → {pressure_score}")
print(f"\nFinal Entertainment Index: {entertainment_score}")

# Show true flip events for verification
flip_events = pbp_df[pbp_df['FLIP']][['PERIOD','PCTIMESTRING','SCOREMARGIN_NUM','SIGN','PREV_SIGN','EVENTMSGTYPE','HOMEDESCRIPTION','VISITORDESCRIPTION']]

display(flip_events)
print("Total true lead-flips:", flip_events.shape[0])

import matplotlib.pyplot as plt

# Recompute star_score to ensure it's up-to-date in this cell:
# If your “star” threshold is 30+ points:
num_star_30 = len(player_stats[player_stats['PTS'] >= 30])
star_score  = 75 * num_star_30

# Or if you meant 40+, swap to:
# num_star_40 = len(player_stats[player_stats['PTS'] >= 40])
# star_score  = 75 * num_star_40

components = {
    'Game Closeness': diff_score,
    'Star 30+ (×75)': star_score,
    'Big Comeback': comeback_score,
    'Overtime': ot_score,
    '1H Lead-Flips': first_half_score,
    '2H Lead-Flips': second_half_score,
    'FGA Attempts': attempts_score,
    'FGA Makes': makes_score,
    'Pressure Makes': pressure_score}

plt.figure(figsize=(10, 6))
bars = plt.barh(list(components.keys()), list(components.values()), color='teal')
plt.title("Component Contributions to Entertainment Index\nDAL vs HOU")
plt.xlabel("Points Contributed")
plt.grid(axis='x', linestyle='--', alpha=0.7)

for bar in bars:
    w = bar.get_width()
    plt.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.0f}", va='center')

plt.tight_layout()
plt.show()

from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2, boxscoresummaryv2, playbyplayv2
import pandas as pd
import time

# Get Dallas Mavericks' team ID
nba_teams = teams.get_teams()
dallas = [team for team in nba_teams if team['full_name'] == 'Dallas Mavericks'][0]
team_name = dallas['abbreviation']
team_id = dallas['id']

# Pull only regular season games for 2023–24
gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id, season_nullable='2023-24')
team_games = gamefinder.get_data_frames()[0]
team_games = team_games[team_games['GAME_ID'].str.startswith('002')]
team_games = team_games.sort_values(by='GAME_DATE', ascending=False)

# Show last 10 games pulled to ensure functionality
print(team_games[['GAME_DATE', 'MATCHUP']].head(10))

# Analyze the last 10 regular season games
entertainment_results = []

for i, row in team_games.head(30).iterrows():
    game_id = row['GAME_ID']
    print(f"Processing game {i + 1}/10 — GAME_ID: {game_id}")

    try:
        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id).get_data_frames()[0]
        summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id).get_data_frames()[5]
        pbp_df = playbyplayv2.PlayByPlayV2(game_id=game_id).get_data_frames()[0]

        team1_row = summary.iloc[0]
        team2_row = summary.iloc[1]

        q4_diff = abs(team1_row['PTS_QTR4'] - team2_row['PTS_QTR4'])

        stars = boxscore[boxscore['PTS'] >= 30]
        num_stars = len(stars)

        half1_team1 = team1_row['PTS_QTR1'] + team1_row['PTS_QTR2']
        half1_team2 = team2_row['PTS_QTR1'] + team2_row['PTS_QTR2']
        comeback = ((half1_team1 < half1_team2 and team1_row['PTS'] > team2_row['PTS']) or
                    (half1_team2 < half1_team1 and team2_row['PTS'] > team1_row['PTS']))

        pbp_df['SCOREMARGIN'] = pd.to_numeric(pbp_df['SCOREMARGIN'], errors='coerce')
        lead_changes = pbp_df['SCOREMARGIN'].dropna().ne(pbp_df['SCOREMARGIN'].shift()).sum()

        clutch_plays = pbp_df[
            (pbp_df['PCTIMESTRING'] <= '05:00') &
            (pbp_df['SCOREMARGIN'].abs() <= 5) &
            (pbp_df['EVENTMSGTYPE'].isin([1, 3]))]
        clutch_pts = 0
        for _, play in clutch_plays.iterrows():
            clutch_pts += 2 if play['EVENTMSGTYPE'] == 1 else 1

        final_fgas = pbp_df[(pbp_df['PCTIMESTRING'] <= '01:00') & (pbp_df['EVENTMSGTYPE'] == 1)]
        final_min_fga = len(final_fgas)

        ot_bonus = 1.5 if 'PTS_OT1' in team1_row and team1_row['PTS_OT1'] > 0 else 0

        index = (
            (1 / (q4_diff + 1)) +
            (0.5 * num_stars) +
            (1 if comeback else 0) +
            ot_bonus +
            (0.3 * lead_changes) +
            (0.5 * clutch_pts) +
            (0.1 * final_min_fga))

        entertainment_results.append({
            'GAME_ID': game_id,
            'DATE': row['GAME_DATE'],
            'OPPONENT': row['MATCHUP'],
            'ENTERTAINMENT_INDEX': index})

        time.sleep(1.2)  # avoid NBA API rate-limiting

    except Exception as e:
        print(f"Skipped {game_id} due to error: {e}")
        continue

# Convert the information to a DataFrame and display first 10
entertainment_df = pd.DataFrame(entertainment_results)
entertainment_df.head(10)

# Line plot
plt.figure(figsize=(12, 6))
plt.plot(pd.to_datetime(entertainment_df['DATE']), entertainment_df['ENTERTAINMENT_INDEX'], marker='o')
plt.title(f"{team_name} Entertainment Index Over 2023–24 Season")
plt.xlabel("Game Date")
plt.ylabel("Entertainment Index")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Histogram
plt.figure(figsize=(8, 5))
plt.hist(entertainment_df['ENTERTAINMENT_INDEX'], bins=15, edgecolor='black')
plt.title("Distribution of Entertainment Scores")
plt.xlabel("Entertainment Index")
plt.ylabel("Number of Games")
plt.tight_layout()
plt.show()

from nba_api.stats.static import teams

# Select teams
team_names = ['Boston Celtics', 'Denver Nuggets', 'Golden State Warriors', 'Milwaukee Bucks', 'Los Angeles Lakers']
nba_teams = teams.get_teams()
team_ids = {team['full_name']: team['id'] for team in nba_teams if team['full_name'] in team_names}
team_ids

from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2, boxscoresummaryv2, playbyplayv2
import pandas as pd
import time

all_entertainment_data = []

# Let user know that the code is processing the team
for team_name, team_id in team_ids.items():
    print(f"Processing {team_name}...")

    # Get games for 2023–24 and filter to regular season
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id, season_nullable='2023-24')
    team_games = gamefinder.get_data_frames()[0]
    team_games = team_games[team_games['GAME_ID'].str.startswith('002')]  # regular season
    team_games = team_games.sort_values(by='GAME_DATE', ascending=False)
    recent_games = team_games.head(30)

    for i, row in recent_games.iterrows():
        game_id = row['GAME_ID']
        print(f"Game {i + 1}/10 — {row['MATCHUP']} on {row['GAME_DATE']}")
        try:
            # Pull data
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id).get_data_frames()[0]
            summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id).get_data_frames()[5]
            pbp_df = playbyplayv2.PlayByPlayV2(game_id=game_id).get_data_frames()[0]


            t1 = summary.iloc[0]
            t2 = summary.iloc[1]

            q4_diff = abs(t1['PTS_QTR4'] - t2['PTS_QTR4'])
            stars = boxscore[boxscore['PTS'] >= 30]
            num_stars = len(stars)

            h1_t1 = t1['PTS_QTR1'] + t1['PTS_QTR2']
            h1_t2 = t2['PTS_QTR1'] + t2['PTS_QTR2']
            comeback = ((h1_t1 < h1_t2 and t1['PTS'] > t2['PTS']) or
                        (h1_t2 < h1_t1 and t2['PTS'] > t1['PTS']))

            pbp_df['SCOREMARGIN'] = pd.to_numeric(pbp_df['SCOREMARGIN'], errors='coerce')
            lead_changes = pbp_df['SCOREMARGIN'].dropna().ne(pbp_df['SCOREMARGIN'].shift()).sum()

            clutch_plays = pbp_df[
                (pbp_df['PCTIMESTRING'] <= '05:00') &
                (pbp_df['SCOREMARGIN'].abs() <= 5) &
                (pbp_df['EVENTMSGTYPE'].isin([1, 3]))]
            clutch_pts = 0
            for _, play in clutch_plays.iterrows():
                clutch_pts += 2 if play['EVENTMSGTYPE'] == 1 else 1

            final_fgas = pbp_df[(pbp_df['PCTIMESTRING'] <= '01:00') & (pbp_df['EVENTMSGTYPE'] == 1)]
            final_min_fga = len(final_fgas)

            ot_bonus = 1.5 if 'PTS_OT1' in t1 and t1.get('PTS_OT1', 0) > 0 else 0

            index = (
                (1 / (q4_diff + 1)) +
                (0.5 * num_stars) +
                (1 if comeback else 0) +
                ot_bonus +
                (0.3 * lead_changes) +
                (0.5 * clutch_pts) +
                (0.1 * final_min_fga))

            all_entertainment_data.append({
                'TEAM': team_name,
                'GAME_ID': game_id,
                'DATE': row['GAME_DATE'],
                'OPPONENT': row['MATCHUP'],
                'ENTERTAINMENT_INDEX': index})

            time.sleep(1.2)

        except Exception as e:
            print(f"Skipped {game_id}: {e}")
            continue


multi_team_df = pd.DataFrame(all_entertainment_data)
# Optional: Convert DATE column to datetime for cleaner formatting
multi_team_df['DATE'] = pd.to_datetime(multi_team_df['DATE'])

# Save to CSV with clean formatting
multi_team_df.to_csv('entertainment_scores_multi_team.csv', index=False)
multi_team_df.head(10)

import matplotlib.pyplot as plt

# Convert date column to datetime
multi_team_df['DATE'] = pd.to_datetime(multi_team_df['DATE'])

multi_team_df.sort_values(by='DATE', inplace=True)

# Line plot per team
plt.figure(figsize=(14, 7))
for team in multi_team_df['TEAM'].unique():
    team_data = multi_team_df[multi_team_df['TEAM'] == team]
    plt.plot(team_data['DATE'], team_data['ENTERTAINMENT_INDEX'], label=team, marker='o', alpha=0.7)

plt.title("Entertainment Index Over the Season by Team")
plt.xlabel("Date")
plt.ylabel("Entertainment Index")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
multi_team_df.boxplot(column='ENTERTAINMENT_INDEX', by='TEAM')
plt.title("Entertainment Score Distribution by Team")
plt.suptitle("")  # Remove default subtitle
plt.xlabel("Team")
plt.ylabel("Entertainment Index")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

top_5 = multi_team_df.sort_values(by='ENTERTAINMENT_INDEX', ascending=False).head(5)
top_5[['TEAM', 'OPPONENT', 'DATE', 'ENTERTAINMENT_INDEX']]

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Apply KMeans with 4 clusters
X = multi_team_df[['ENTERTAINMENT_INDEX']].copy()
kmeans = KMeans(n_clusters=4, random_state=42)
multi_team_df['CLUSTER'] = kmeans.fit_predict(X)
cluster_centers = kmeans.cluster_centers_.flatten()
labels = ['Low', 'Medium', 'High', 'Really High']  # 4 tiers

# Sort cluster centers to assign labels in ascending order
order = pd.Series(cluster_centers).sort_values().index.tolist()
label_mapping = {order[i]: labels[i] for i in range(4)}
multi_team_df['ENTERTAINMENT_LEVEL'] = multi_team_df['CLUSTER'].map(label_mapping)

multi_team_df.to_csv("entertainment_scores_with_4clusters.csv", index=False)

# Plotting
ent_summary = multi_team_df.groupby(['TEAM', 'ENTERTAINMENT_LEVEL']).size().unstack().fillna(0)
ent_summary.plot(
    kind='bar',
    stacked=True,
    figsize=(12, 6),
    colormap='tab20')

plt.title("Number of Games by Entertainment Tier per Team (4 Clusters)")
plt.xlabel("Team")
plt.ylabel("Number of Games")
plt.legend(title="Tier")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Show thresholds and centroids
centroids = pd.Series(cluster_centers, index=[f"Cluster {i}" for i in range(4)])
centroids_sorted = centroids.sort_values()

print("Cluster Centroids (Entertainment Index Averages):")
for i, val in enumerate(centroids_sorted):
    label = labels[i]
    print(f"{label} Entertainment: {val:.2f}")

# Optional: Show value ranges in each cluster
for label in ['Low', 'Medium', 'High', 'Really High']:
    cluster_vals = multi_team_df[multi_team_df['ENTERTAINMENT_LEVEL'] == label]['ENTERTAINMENT_INDEX']
    print(f"\n{label} Entertainment Range:")
    print(f"Min: {cluster_vals.min():.2f}")
    print(f"Max: {cluster_vals.max():.2f}")
    print(f"Mean: {cluster_vals.mean():.2f}")

