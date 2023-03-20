import requests
import pandas as pd


r =requests.get('https://financialmodelingprep.com/api/v3/historical-chart/60min/AAPL?apikey=3e17d2b777a13feee4c1243985cdc7c4')


r = r.text

print(r)

