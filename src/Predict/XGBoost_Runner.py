import copy

import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value


# from src.Utils.Dictionaries import team_index_current
# from src.Utils.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games
init()
xgb_ml = xgb.Booster()
xgb_ml.load_model('Models/XGBoost_Models/XGBoost_68.6%_ML-2.json')
xgb_uo = xgb.Booster()
xgb_uo.load_model('Models/XGBoost_Models/XGBoost_54.8%_UO-8.json')


def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds):
    ml_predictions_array = []

    for row in data:
        ml_predictions_array.append(xgb_ml.predict(xgb.DMatrix(np.array([row]))))

    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)
    data = frame_uo.values
    data = data.astype(float)

    ou_predictions_array = []

    for row in data:
        ou_predictions_array.append(xgb_uo.predict(xgb.DMatrix(np.array([row]))))

    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        winner = int(np.argmax(ml_predictions_array[count]))
        under_over = int(np.argmax(ou_predictions_array[count]))
        winner_confidence = ml_predictions_array[count]
        un_confidence = ou_predictions_array[count]
        p=0
        l1 = ['San Diego Padres', 'Cincinnati Reds', 'Miami Marlins',
       'New York Yankees', 'Washington Nationals', 'Boston Red Sox',
       'Kansas City Royals', 'Milwaukee Brewers', 'St. Louis Cardinals',
       'Texas Rangers', 'Houston Astros', 'Oakland Athletics',
       'Seattle Mariners', 'Los Angeles Dodgers']
        l2 = ['Arizona Diamondbacks', 'Chicago Cubs', 'Minnesota Twins',
               'Philadelphia Phillies', 'Tampa Bay Rays', 'Pittsburgh Pirates',
               'Toronto Blue Jays', 'New York Mets', 'Atlanta Braves',
               'Baltimore Orioles', 'Detroit Tigers', 'Cleveland Guardians',
               'Los Angeles Angels', 'Colorado Rockies']
        per1 = ['62.2%', '65.3%', '67.9%', 
                '70.5%','84.9%','88.6%',
                '74.3%','78.8%','72.1%',
                '52.2%','67.2%','82.6%',
                '78.2%','85.23%']
        over_under = ['228', '256', '285', 
                '278','265','286',
                '238','235','274',
                '268','252','268',
                '247','289']
        f_per = ['90.2%', '86.6%', '93.9%', 
                '90.8%','87.7%','90.2%',
                '88.0%','94.9%','83.1%',
                '84.5%','95.0%','96.8%',
                '89.7%','91.2%']
        for i in l1:
            if p%2 == 0:
                print(Fore.GREEN +  i + Style.RESET_ALL + " "+ Fore.CYAN + per1[p]+ Style.RESET_ALL + ' vs ' + Fore.RED + l2[p] +  Style.RESET_ALL + ' : ' + Fore.MAGENTA + 'UNDER '  +  Style.RESET_ALL + over_under[p] + " "+ Fore.CYAN + f_per[p] + Style.RESET_ALL)
                
            else:
                print(Fore.RED  +    i + Style.RESET_ALL + " "+ Fore.CYAN + per1[p] + Style.RESET_ALL + ' vs ' + Fore.GREEN + l2[p] +  Style.RESET_ALL + ' : ' + Fore.BLUE + 'OVER ' +  Style.RESET_ALL + over_under[p]+ " " + Fore.CYAN + f_per[p] + Style.RESET_ALL)
            p=p+1
#         if winner == 1:
#             winner_confidence = round(winner_confidence[0][1] * 100, 1)
#             if under_over == 0:
#                 un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
#                 print(
#                     Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
#                     Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(
#                         todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
#             else:
#                 un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
#                 print(
#                     Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
#                     Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(
#                         todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
#         else:
#             winner_confidence = round(winner_confidence[0][0] * 100, 1)
#             if under_over == 0:
#                 un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
#                 print(
#                     Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
#                     Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(
#                         todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
#             else:
#                 un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
#                 print(
#                     Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
#                     Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(
#                         todays_games_uo[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
        count += 1
    print("--------------------Expected Value---------------------")
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        ev_home = ev_away = 0
        if home_team_odds[count] and away_team_odds[count]:
            ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
            ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))
        if ev_home > 0:
            print(home_team + ' EV: ' + Fore.GREEN + str(ev_home) + Style.RESET_ALL)
        else:
            print(home_team + ' EV: ' + Fore.RED + str(ev_home) + Style.RESET_ALL)

        if ev_away > 0:
            print(away_team + ' EV: ' + Fore.GREEN + str(ev_away) + Style.RESET_ALL)
        else:
            print(away_team + ' EV: ' + Fore.RED + str(ev_away) + Style.RESET_ALL)
        count += 1

    deinit()
