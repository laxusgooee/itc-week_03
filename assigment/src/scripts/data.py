import pandas as pd
from settings import BASE_PATH

CSV_PATH = BASE_PATH  / "data" / "fifa_player_performance_market_value.csv"

df = pd.read_csv(CSV_PATH)