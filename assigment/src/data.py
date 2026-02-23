import pandas as pd
from settings import BASE_PATH


df = pd.read_csv(BASE_PATH  / "data" / "fifa_player_performance_market_value.csv")

df_time_series = pd.read_csv(BASE_PATH  / "data" / "online_retail_II.csv")