# Auto-generated Colab file

# Pranav's code

# Setup

# !pip install nba_api

from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2, boxscoresummaryv2
import pandas as pd
import time

# Find OT game

# Get games
nba_games = leaguegamefinder.LeagueGameFinder(season_nullable='2023-24', league_id_nullable='00')
games_df = nba_games.get_data_frames()[0]

# Filter OT games - 240+ mins = OT
ot_games = games_df[games_df['MIN'] > 240]
my_game = ot_games.iloc[0] # Just grab first one
game_id = my_game['GAME_ID']

# Game info
my_game[['GAME_DATE', 'MATCHUP', 'TEAM_NAME', 'PTS']]

# Get box score data

# Get stats
box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
summ = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)

# Player stats
player_stats = box.get_data_frames()[0]

# Quarter scores
line_score = summ.get_data_frames()[5]  # line score is at idx 5

# Handle OT
if 'PTS_OT1' not in line_score.columns:
    line_score['PTS_OT1'] = 0  # some games don't have OT

# Show scoring
line_score[['TEAM_ABBREVIATION', 'PTS_QTR1', 'PTS_QTR2', 'PTS_QTR3', 'PTS_QTR4', 'PTS_OT1', 'PTS']]

# Extract team data to use later
team1 = line_score.iloc[0]
team2 = line_score.iloc[1] 

# Calculate Entertainment Index

from nba_api.stats.endpoints import playbyplayv2
import pandas as pd
import numpy as np  # needed for NaN stuff

# Process play-by-play
pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
pbp_df = pbp.get_data_frames()[0]
pbp_df['SCOREMARGIN'] = pd.to_numeric(pbp_df['SCOREMARGIN'], errors='coerce')  # handle non-numeric values
# Convert time format to seconds remaining
pbp_df['SECONDS_LEFT'] = (
    pbp_df['PCTIMESTRING'].str.split(':').str[0].astype(float) * 60 +
    pbp_df['PCTIMESTRING'].str.split(':').str[1].astype(float)
)

# Score diff bonus
final_diff = abs(team1['PTS'] - team2['PTS'])
if final_diff <= 5:
    diff_score = 95  # super close game
elif final_diff <= 10:
    diff_score = 85  # pretty close
elif final_diff <= 20:
    diff_score = 30  # not that close
else:
    diff_score = 0   # blowout

# Star player bonus
stars_30pts = len(player_stats[player_stats['PTS'] >= 30])
star_bonus = 75 * stars_30pts

# Comeback bonus
half1_t1 = team1['PTS_QTR1'] + team1['PTS_QTR2']
half1_t2 = team2['PTS_QTR1'] + team2['PTS_QTR2']
halftime_diff = abs(half1_t1 - half1_t2)
leader_half = 'team1' if half1_t1 > half1_t2 else 'team2'
winner = 'team1' if team1['PTS'] > team2['PTS'] else 'team2'
big_comeback = (halftime_diff >= 20 and leader_half != winner)
comeback_bonus = 88 if big_comeback else 0

# OT bonus
ot_bonus = 98 if (team1.get('PTS_OT1', 0) > 0 or team2.get('PTS_OT1', 0) > 0) else 0

# Lead changes
pbp_df['SCOREMARGIN_NUM'] = pbp_df['SCOREMARGIN'] 
pbp_df['SIGN'] = pbp_df['SCOREMARGIN_NUM'].apply(lambda x: 1 if x>0 else (-1 if x<0 else np.nan))
pbp_df['PREV_SIGN'] = pbp_df['SIGN'].shift()
pbp_df['FLIP'] = (pbp_df['SIGN'] * pbp_df['PREV_SIGN'] == -1)
pbp_df['HALF'] = pbp_df['PERIOD'].apply(lambda p: 1 if p<=2 else 2)

# count flips by half
h1_flips = int(pbp_df[(pbp_df['HALF']==1) & pbp_df['FLIP']].shape[0])
h2_flips = int(pbp_df[(pbp_df['HALF']==2) & pbp_df['FLIP']].shape[0])
h1_flip_score = 15 * h1_flips
h2_flip_score = 25 * h2_flips

# Final-min shots
fga_plays = pbp_df[
    (pbp_df['PERIOD']==4) &
    (pbp_df['SECONDS_LEFT']<=300) &  # last 5 min
    (pbp_df['EVENTMSGTYPE']==1) &    # field goal attempt
    (pbp_df['SCOREMARGIN_NUM'].abs()<10)  # close game
]
attempts = len(fga_plays)
makes = 0
for idx, row in fga_plays.iterrows():
    old_margin = row['SCOREMARGIN_NUM']
    new_margin = pbp_df.loc[idx+1:, 'SCOREMARGIN_NUM'].dropna().iloc[0]
    if abs(new_margin) != abs(old_margin):  # score changed
        makes += 1
attempt_pts = 15 * attempts
make_pts = 30 * makes

# Pressure makes - shots when game is close
pressure_makes = pbp_df[
    (pbp_df['EVENTMSGTYPE'].isin([1,3])) &  # FG or FT
    (pbp_df['SCOREMARGIN_NUM'].abs() < 10)   # close game
].shape[0]
pressure_pts = 10 * pressure_makes

# Add everything up
excitement_score = (
    diff_score + star_bonus + comeback_bonus + ot_bonus +
    h1_flip_score + h2_flip_score +
    attempt_pts + make_pts +
    pressure_pts
)

# Print results
print("Final Metric Breakdown (0-100+ scale):")
print(f"Total-Game Close Bonus   > {diff_score}  (margin={final_diff})")
print(f"Star 30+ (x75 each)      > {star_bonus}  ({stars_30pts} player(s))")
print(f"Big Comeback (>=20)      > {comeback_bonus}")
print(f"Overtime Bonus           > {ot_bonus}")
print(f"1H Lead-Flips (x15)      > {h1_flips} -> {h1_flip_score}")
print(f"2H Lead-Flips (x25)      > {h2_flips} -> {h2_flip_score}")
print(f"FGA Attempts (x15)       > {attempts} -> {attempt_pts}")
print(f"Makes in those FGAs (x30)> {makes} -> {make_pts}")
print(f"Pressure Makes (x10)     > {pressure_makes} -> {pressure_pts}")
print(f"\nFinal Entertainment Index: {excitement_score}")

# Show lead flips
flip_events = pbp_df[pbp_df['FLIP']][[
    'PERIOD','PCTIMESTRING','SCOREMARGIN_NUM','SIGN','PREV_SIGN',
    'EVENTMSGTYPE','HOMEDESCRIPTION','VISITORDESCRIPTION'
]]
print(flip_events)
print("Total true lead-flips:", flip_events.shape[0])

# Visualization

import matplotlib.pyplot as plt

# Recalc star score - need to do this again for some reason
stars_30pts = len(player_stats[player_stats['PTS'] >= 30])
star_bonus = 75 * stars_30pts

# Dict of all components for bar chart
score_comps = {
    'Game Closeness':    diff_score,
    'Star 30+ (x75)':    star_bonus,
    'Big Comeback':      comeback_bonus,
    'Overtime':          ot_bonus,
    '1H Lead-Flips':     h1_flip_score,
    '2H Lead-Flips':     h2_flip_score,
    'FGA Attempts':      attempt_pts,
    'FGA Makes':         make_pts,
    'Pressure Makes':    pressure_pts
}

plt.figure(figsize=(10, 6))
bars = plt.barh(list(score_comps.keys()), list(score_comps.values()), color='teal')
plt.title("Component Contributions to Entertainment Index\nBOS vs DAL")
plt.xlabel("Points Contributed")
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add text labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.0f}", va='center')

plt.tight_layout()
plt.show()

# Full season analysis

from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2, boxscoresummaryv2, playbyplayv2
import pandas as pd
import time

# Get Mavs ID
teams_list = teams.get_teams()
mavs = [team for team in teams_list if team['full_name'] == 'Dallas Mavericks'][0]
team_abbr = mavs['abbreviation']
team_id = mavs['id']

# Get reg season games
mavs_games = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id, season_nullable='2023-24')
games = mavs_games.get_data_frames()[0]
games = games[games['GAME_ID'].str.startswith('002')]  # just reg season
games = games.sort_values(by='GAME_DATE', ascending=False)

# Show games
print(games[['GAME_DATE', 'MATCHUP']].head(30))

# Analyze recent games
results = []

for i, row in games.head(30).iterrows():  # Looking at 30 most recent games
    gid = row['GAME_ID']
    print(f"Processing game {i + 1}/10 — GAME_ID: {gid}")

    try:
        # Get the data
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=gid).get_data_frames()[0]
        summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=gid).get_data_frames()[5]
        pbp = playbyplayv2.PlayByPlayV2(game_id=gid).get_data_frames()[0]

        # Get team data
        team1 = summary.iloc[0]
        team2 = summary.iloc[1]

        # Calculate various metrics
        q4_diff = abs(team1['PTS_QTR4'] - team2['PTS_QTR4'])
        
        stars = box[box['PTS'] >= 30]
        num_stars = len(stars)

        # Check for comeback
        half1_t1 = team1['PTS_QTR1'] + team1['PTS_QTR2']
        half1_t2 = team2['PTS_QTR1'] + team2['PTS_QTR2']
        comeback = ((half1_t1 < half1_t2 and team1['PTS'] > team2['PTS']) or
                    (half1_t2 < half1_t1 and team2['PTS'] > team1['PTS']))

        # Prepare PBP data
        pbp['SCOREMARGIN'] = pd.to_numeric(pbp['SCOREMARGIN'], errors='coerce')
        lead_changes = pbp['SCOREMARGIN'].dropna().ne(pbp['SCOREMARGIN'].shift()).sum()

        # Find clutch plays
        clutch_plays = pbp[
            (pbp['PCTIMESTRING'] <= '05:00') &
            (pbp['SCOREMARGIN'].abs() <= 5) &
            (pbp['EVENTMSGTYPE'].isin([1, 3]))
        ]
        clutch_pts = 0
        for _, play in clutch_plays.iterrows():
            clutch_pts += 2 if play['EVENTMSGTYPE'] == 1 else 1

        # Last minute FGAs
        final_fgas = pbp[
            (pbp['PCTIMESTRING'] <= '01:00') &
            (pbp['EVENTMSGTYPE'] == 1)
        ]
        final_min_fga = len(final_fgas)

        # Overtime check
        went_to_ot = 'PTS_OT1' in team1 and team1['PTS_OT1'] > 0
        ot_bonus = 1.5 if went_to_ot else 0

        # Calculate final index
        index_score = (
            (1 / (q4_diff + 1)) +   # avoid div by zero
            (0.5 * num_stars) +
            (1 if comeback else 0) +
            ot_bonus +
            (0.3 * lead_changes) +
            (0.5 * clutch_pts) +
            (0.1 * final_min_fga)
        )

        # Add to results
        results.append({
            'GAME_ID': gid,
            'DATE': row['GAME_DATE'],
            'OPPONENT': row['MATCHUP'],
            'ENTERTAINMENT_INDEX': index_score
        })

        time.sleep(1.2)  # avoid API limits

    except Exception as e:
        print(f"Skipped {gid} due to error: {e}")
        continue

# Create dataframe
df = pd.DataFrame(results)
df.head(10)

# Visualize season

# Line plot
plt.figure(figsize=(12, 6))
plt.plot(pd.to_datetime(df['DATE']), df['ENTERTAINMENT_INDEX'], marker='o')
plt.title(f"{team_abbr} Entertainment Index Over 2023-24 Season")
plt.xlabel("Game Date")
plt.ylabel("Entertainment Index")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Histogram
plt.figure(figsize=(8, 5))
plt.hist(df['ENTERTAINMENT_INDEX'], bins=15, edgecolor='black')
plt.title("Distribution of Entertainment Scores")
plt.xlabel("Entertainment Index")
plt.ylabel("Number of Games")
plt.tight_layout()
plt.show()

# Multi-team comparison

from nba_api.stats.static import teams

# Select top teams
my_teams = ['Boston Celtics', 'Denver Nuggets', 'Golden State Warriors', 'Milwaukee Bucks', 'Los Angeles Lakers']
teams_list = teams.get_teams()
team_ids = {team['full_name']: team['id'] for team in teams_list if team['full_name'] in my_teams}
team_ids

# Multi-team analysis

from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2, boxscoresummaryv2, playbyplayv2
import pandas as pd
import time

all_data = []

for team_name, team_id in team_ids.items():
    print(f"\nProcessing {team_name}...")

    # Get reg season
    games = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id, season_nullable='2023-24')
    team_games = games.get_data_frames()[0]
    team_games = team_games[team_games['GAME_ID'].str.startswith('002')]
    team_games = team_games.sort_values(by='GAME_DATE', ascending=False)
    recent_games = team_games.head(30)  # get 30 most recent games

    for i, row in recent_games.iterrows():
        gid = row['GAME_ID']
        print(f"  Game {i + 1}/10 — {row['MATCHUP']} on {row['GAME_DATE']}")
        try:
            # Pull data
            box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=gid).get_data_frames()[0]
            summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=gid).get_data_frames()[5]
            pbp = playbyplayv2.PlayByPlayV2(game_id=gid).get_data_frames()[0]

            # Teams
            home = summary.iloc[0]
            away = summary.iloc[1]

            # Calculations
            q4_diff = abs(home['PTS_QTR4'] - away['PTS_QTR4'])
            stars = box[box['PTS'] >= 30]
            num_stars = len(stars)

            h1_home = home['PTS_QTR1'] + home['PTS_QTR2']
            h1_away = away['PTS_QTR1'] + away['PTS_QTR2']
            comeback = ((h1_home < h1_away and home['PTS'] > away['PTS']) or
                        (h1_away < h1_home and away['PTS'] > home['PTS']))

            pbp['SCOREMARGIN'] = pd.to_numeric(pbp['SCOREMARGIN'], errors='coerce')
            lead_changes = pbp['SCOREMARGIN'].dropna().ne(pbp['SCOREMARGIN'].shift()).sum()

            clutch_plays = pbp[
                (pbp['PCTIMESTRING'] <= '05:00') &
                (pbp['SCOREMARGIN'].abs() <= 5) &
                (pbp['EVENTMSGTYPE'].isin([1, 3]))
            ]
            clutch_pts = 0
            for _, play in clutch_plays.iterrows():
                clutch_pts += 2 if play['EVENTMSGTYPE'] == 1 else 1

            final_fgas = pbp[(pbp['PCTIMESTRING'] <= '01:00') & (pbp['EVENTMSGTYPE'] == 1)]
            final_min_fga = len(final_fgas)

            # Check for OT
            had_ot = 'PTS_OT1' in home and home.get('PTS_OT1', 0) > 0
            ot_bonus = 1.5 if had_ot else 0

            # Final calc
            excitement = (
                (1 / (q4_diff + 1)) +
                (0.5 * num_stars) +
                (1 if comeback else 0) +
                ot_bonus +
                (0.3 * lead_changes) +
                (0.5 * clutch_pts) +
                (0.1 * final_min_fga)
            )

            all_data.append({
                'TEAM': team_name,
                'GAME_ID': gid,
                'DATE': row['GAME_DATE'],
                'OPPONENT': row['MATCHUP'],
                'ENTERTAINMENT_INDEX': excitement
            })

            time.sleep(1.2)  # API rate limiting

        except Exception as e:
            print(f"  Skipped {gid}: {e}")
            continue

# Put it all together
team_df = pd.DataFrame(all_data)
team_df['DATE'] = pd.to_datetime(team_df['DATE'])

# Save to CSV
team_df.to_csv('entertainment_scores_multi_team.csv', index=False)
team_df.head(10)

# Visualize teams

import matplotlib.pyplot as plt

# Fix dates
team_df['DATE'] = pd.to_datetime(team_df['DATE'])

# Sort chronologically
team_df.sort_values(by='DATE', inplace=True)

# Line plot per team
plt.figure(figsize=(14, 7))
for team in team_df['TEAM'].unique():
    team_data = team_df[team_df['TEAM'] == team]
    plt.plot(team_data['DATE'], team_data['ENTERTAINMENT_INDEX'], label=team, marker='o', alpha=0.7)

plt.title("Entertainment Index Over the Season by Team")
plt.xlabel("Date")
plt.ylabel("Entertainment Index")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Box plot
plt.figure(figsize=(10, 6))
team_df.boxplot(column='ENTERTAINMENT_INDEX', by='TEAM')
plt.title("Entertainment Score Distribution by Team")
plt.suptitle("")  # Remove default subtitle
plt.xlabel("Team")
plt.ylabel("Entertainment Index")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Find top games
top_5 = team_df.sort_values(by='ENTERTAINMENT_INDEX', ascending=False).head(5)
top_5[['TEAM', 'OPPONENT', 'DATE', 'ENTERTAINMENT_INDEX']]

# Categorize entertainment - using quantiles instead of KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Use quantiles to categorize entertainment scores
# This approach is simpler and avoids issues with KMeans
team_df['ENTERTAINMENT_LEVEL'] = pd.qcut(
    team_df['ENTERTAINMENT_INDEX'], 
    q=4, 
    labels=['Low', 'Medium', 'High', 'Really High']
)

# Save the results
team_df.to_csv("entertainment_scores_with_quantiles.csv", index=False)

# Calculate thresholds (quantile values)
quantile_values = team_df['ENTERTAINMENT_INDEX'].quantile([0.25, 0.5, 0.75])
print("Entertainment Index Thresholds:")
print(f"Low: Below {quantile_values[0.25]:.2f}")
print(f"Medium: {quantile_values[0.25]:.2f} to {quantile_values[0.5]:.2f}")
print(f"High: {quantile_values[0.5]:.2f} to {quantile_values[0.75]:.2f}")
print(f"Really High: Above {quantile_values[0.75]:.2f}")

# Stacked bar chart for entertainment tiers
team_summary = team_df.groupby(['TEAM', 'ENTERTAINMENT_LEVEL']).size().unstack().fillna(0)

team_summary.plot(
    kind='bar',
    stacked=True,
    figsize=(12, 6),
    colormap='tab20'
)
plt.title("Number of Games by Entertainment Tier per Team (Quantile-Based)")
plt.xlabel("Team")
plt.ylabel("Number of Games")
plt.legend(title="Tier")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Show value ranges for each category
for cat in ['Low', 'Medium', 'High', 'Really High']:
    group_vals = team_df[team_df['ENTERTAINMENT_LEVEL'] == cat]['ENTERTAINMENT_INDEX']
    print(f"\n{cat} Entertainment Range:")
    print(f"Min: {group_vals.min():.2f}")
    print(f"Max: {group_vals.max():.2f}")
    print(f"Mean: {group_vals.mean():.2f}")

