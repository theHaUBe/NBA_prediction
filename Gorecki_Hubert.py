import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import json
import sys

#przygotowanie danych do wytrenowania modelu
#odrzucenie kolumn, mapowanie kolumn, tworzenie modelu
def train_and_save_model(data_path, drop_columns, target_column, mapping, model_filename, additional_mapping=None):
    data = pd.read_csv(data_path)

    data[target_column] = data[target_column].map(mapping)

    if additional_mapping:
        for column, map_dict in additional_mapping.items():
            data[column] = data[column].map(map_dict)

    data = data.dropna()
    data = data.drop(columns=drop_columns)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.000001, random_state=42)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    joblib.dump(model, model_filename)


#tworzenie dwoch modeli dla ALL-NBA oraz ROOKIES
train_and_save_model(
    data_path='all_stats.csv',
    drop_columns=['Player', 'Year', 'TOV'],
    target_column='ALL_NBA',
    mapping={'First': 1, 'Second': 2, 'Third': 3, 'not': 4},
    model_filename='model_all_nba.joblib',
    additional_mapping={'Position': {'F': 1, 'G': 2, 'C': 3}}
)

train_and_save_model(
    data_path='all_rookies.csv',
    drop_columns=['Player', 'Year'],
    target_column='Team',
    mapping={'First': 1, 'Second': 2, 'not': 3},
    model_filename='model_rookies_nba.joblib'
)


def predict_and_generate_json(output_filename):

    # Predykcja dla ROOKIES
    data_rookies = pd.read_csv('2023_rookies.csv')
    clf_rookies = joblib.load('model_rookies_nba.joblib')

    players_rookies = data_rookies['Player']

    X_pred_rookies = data_rookies.drop(columns=['Player'])
    scaler = StandardScaler()
    X_pred_rookies_scaled = scaler.fit_transform(X_pred_rookies)

    predictions_rookies = clf_rookies.predict(X_pred_rookies_scaled)

    inv_all_nba_mapping = {1: 'First', 2: 'Second'}
    predicted_teams_rookies = [inv_all_nba_mapping.get(pred, 'not') for pred in predictions_rookies]

    all_nba_teams = {
        "first all-nba team": [],
        "second all-nba team": [],
        "third all-nba team": [],
        "first rookie all-nba team": [],
        "second rookie all-nba team": []
    }

    for i, predicted_team in enumerate(predicted_teams_rookies):
        if predicted_team in ['First', 'Second']:
            all_nba_teams[predicted_team.lower() + " rookie all-nba team"].append(
                (players_rookies.iloc[i], data_rookies.iloc[i]['PTS'], data_rookies.iloc[i]['FGM'],
                 data_rookies.iloc[i]['FGA'])
            )

    all_nba_teams["first rookie all-nba team"] = sorted(all_nba_teams["first rookie all-nba team"], key=lambda x: x[1],
                                                        reverse=True)

    first_team_rookies = all_nba_teams["first rookie all-nba team"]
    if len(first_team_rookies) > 5:
        all_nba_teams["second rookie all-nba team"].extend(first_team_rookies[5:])
        all_nba_teams["first rookie all-nba team"] = first_team_rookies[:5]

    all_nba_teams["second rookie all-nba team"] = sorted(all_nba_teams["second rookie all-nba team"],
                                                         key=lambda x: x[2] / x[3], reverse=True)[:5]

    for team in all_nba_teams:
        if 'rookie' in team:
            all_nba_teams[team] = [player for player, pts, fgm, fga in all_nba_teams[team]]

    # Predykcja dla All-NBA
    data_all_nba = pd.read_csv('2023_season.csv')
    clf_all_nba = joblib.load('model_all_nba.joblib')

    position_mapping = {'F': 1, 'G': 2, 'C': 3}
    inv_position_mapping = {v: k for k, v in position_mapping.items()}
    data_all_nba['Position'] = data_all_nba['Position'].map(position_mapping)
    data_all_nba = data_all_nba.reset_index(drop=True)

    players_all_nba = data_all_nba['Player']
    points_all_nba = data_all_nba['PTS']

    X_pred_all_nba = data_all_nba.drop(columns=['Player', 'TOV'])
    scaler = StandardScaler()
    X_pred_all_nba_scaled = scaler.fit_transform(X_pred_all_nba)

    predictions_all_nba = clf_all_nba.predict(X_pred_all_nba_scaled)

    inv_all_nba_mapping = {1: 'First', 2: 'Second', 3: 'Third', 4: 'not'}
    predicted_teams_all_nba = [inv_all_nba_mapping[pred] for pred in predictions_all_nba]

    for i, predicted_team in enumerate(predicted_teams_all_nba):
        if predicted_team in ['First', 'Second', 'Third']:
            all_nba_teams[predicted_team.lower() + " all-nba team"].append(
                (players_all_nba.iloc[i], data_all_nba.iloc[i]['Position'], points_all_nba.iloc[i])
            )

    def select_team(players):
        team = []
        forwards = [p for p in players if p[1] == position_mapping['F']]
        guards = [p for p in players if p[1] == position_mapping['G']]
        centers = [p for p in players if p[1] == position_mapping['C']]

        forwards = sorted(forwards, key=lambda x: x[2], reverse=True)[:2]
        guards = sorted(guards, key=lambda x: x[2], reverse=True)[:2]
        centers = sorted(centers, key=lambda x: x[2], reverse=True)[:1]

        team.extend(forwards)
        team.extend(guards)
        team.extend(centers)

        return team

    for team, players in all_nba_teams.items():
        if 'rookie' not in team:
            all_nba_teams[team] = select_team(players)

    for team, players in all_nba_teams.items():
        if 'rookie' not in team:
            all_nba_teams[team] = [{"Player": player, "Position": inv_position_mapping[position], "PTS": int(pts)} for
                                   player, position, pts in players]

    for team, players in all_nba_teams.items():
        if 'rookie' not in team:
            all_nba_teams[team] = [player["Player"] for player in players]

    with open(output_filename, 'w', encoding='utf-8') as json_file:
        json.dump(all_nba_teams, json_file, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py output_filename.json")
        sys.exit(1)

    output_filename = sys.argv[1]
    predict_and_generate_json(output_filename)
