import pandas as pd
from euroleague_api.boxscore_data import BoxScoreData
import logging

# Nustatome konkurso kodą ir sezono metus
competition = "E"  # Naudoti "U" Eurocup turnyrui
season = 2024  # Naudoti 2024 metus

# Inicializuojame BoxScoreData klasę su nurodytu konkursu
boxscoredata = BoxScoreData(competition=competition)

# Sukuriame tuščią DataFrame visiems surinktiems duomenims
all_data = []

# Iteruojame per game_code nuo 1 iki tol, kol rasime duomenis
failed_attempts = 0
for game_code in range(1, 1000):  # 1000 - spėtinis maksimalus game_code skaičius
    try:
        # Gauname pasirinkto gamecode rungtynių duomenis
        boxscore_stats_df = boxscoredata.get_player_boxscore_stats_data(season=season, gamecode=game_code)
        boxscore_stats_df['gamecode'] = game_code  # Pridedame gamecode stulpelį
        all_data.append(boxscore_stats_df)
        print(f"Duomenys gauti game_code: {game_code}")
    except Exception as e:
        logging.error(f"Failed to get stats for game code {game_code}: {e}")
        failed_attempts += 1
        if failed_attempts > 10:
            print("Daugiau nei 10 rungtynių nepavyko gauti, nutraukiama seka.")
            break
        continue

# Jei surinkome duomenis, sujungiame juos į vieną DataFrame
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    # Išsaugome visus duomenis į failą
    final_df.to_csv('data/euroleague_boxscores_all.csv', index=False)
    print("Visi surinkti duomenys išsaugoti 'euroleague_boxscores_all.csv' faile.")
else:
    print("Nepavyko surinkti jokių duomenų.")