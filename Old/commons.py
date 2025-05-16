from datetime import datetime, timedelta

today = datetime.now()
week = today - timedelta(days=7)
two_week = today - timedelta(days=7)
month = today - timedelta(days=30)

TODAY_DASHED = today.strftime("%Y-%m-%d")
TODAY_SLASHED = today.strftime("%m/%d/%Y")
WEEK_DASHES = week.strftime("%Y-%m-%d")
TWO_WEEK_DASHED = two_week.strftime("%Y-%m-%d")
MONTH_DASHED = month.strftime("%Y-%m-%d")

MLB_TEAMS = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres",
    "SEA": "Seattle Mariners",
    "SF": "San Francisco Giants",
    "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals"
}