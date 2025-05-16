import requests
from datetime import datetime

class BallparkWeather:
    def __init__(self, api_key):
        self.api_key = api_key

        # Team → Stadium info
        self.team_ballparks = {
            "ARI": {"stadium": "Chase Field", "lat": 33.4455, "lon": -112.0667},
            "ATL": {"stadium": "Truist Park", "lat": 33.8908, "lon": -84.4678},
            "BAL": {"stadium": "Camden Yards", "lat": 39.2839, "lon": -76.6218},
            "BOS": {"stadium": "Fenway Park", "lat": 42.3467, "lon": -71.0972},
            "CHC": {"stadium": "Wrigley Field", "lat": 41.9484, "lon": -87.6553},
            "CIN": {"stadium": "Great American Ball Park", "lat": 39.0972, "lon": -84.5072},
            "CLE": {"stadium": "Progressive Field", "lat": 41.4962, "lon": -81.6852},
            "COL": {"stadium": "Coors Field", "lat": 39.7559, "lon": -104.9942},
            "CWS": {"stadium": "Guaranteed Rate Field", "lat": 41.8299, "lon": -87.6339},
            "DET": {"stadium": "Comerica Park", "lat": 42.3390, "lon": -83.0490},
            "HOU": {"stadium": "Minute Maid Park", "lat": 29.7572, "lon": -95.3556},
            "KC": {"stadium": "Kauffman Stadium", "lat": 39.0516, "lon": -94.4804},
            "LAA": {"stadium": "Angel Stadium", "lat": 33.8003, "lon": -117.8827},
            "LAD": {"stadium": "Dodger Stadium", "lat": 34.0739, "lon": -118.2400},
            "MIA": {"stadium": "LoanDepot Park", "lat": 25.7780, "lon": -80.2195},
            "MIL": {"stadium": "American Family Field", "lat": 43.0280, "lon": -87.9711},
            "MIN": {"stadium": "Target Field", "lat": 44.9817, "lon": -93.2789},
            "NYM": {"stadium": "Citi Field", "lat": 40.7571, "lon": -73.8458},
            "NYY": {"stadium": "Yankee Stadium", "lat": 40.8296, "lon": -73.9262},
            "OAK": {"stadium": "Oakland Coliseum", "lat": 37.7516, "lon": -122.2005},
            "PHI": {"stadium": "Citizens Bank Park", "lat": 39.9061, "lon": -75.1665},
            "PIT": {"stadium": "PNC Park", "lat": 40.4469, "lon": -80.0057},
            "SD": {"stadium": "Petco Park", "lat": 32.7073, "lon": -117.1573},
            "SEA": {"stadium": "T-Mobile Park", "lat": 47.5914, "lon": -122.3325},
            "SF": {"stadium": "Oracle Park", "lat": 37.7786, "lon": -122.3893},
            "STL": {"stadium": "Busch Stadium", "lat": 38.6226, "lon": -90.1928},
            "TB": {"stadium": "Tropicana Field", "lat": 27.7683, "lon": -82.6534},
            "TEX": {"stadium": "Globe Life Field", "lat": 32.7473, "lon": -97.0847},
            "TOR": {"stadium": "Rogers Centre", "lat": 43.6414, "lon": -79.3894},
            "WSH": {"stadium": "Nationals Park", "lat": 38.8728, "lon": -77.0075},
        }

    def get_home_team(self, matchup_str):
        parts = matchup_str.strip().upper().split("VS")
        if len(parts) != 2:
            raise ValueError("Matchup format must be like 'NYY vs SEA'")
        return parts[1].strip()

    def get_forecast(self, matchup_str, game_hour=19, game_date=None):
        home_team = self.get_home_team(matchup_str)
        if home_team not in self.team_ballparks:
            raise ValueError(f"Unknown team code: {home_team}")

        park = self.team_ballparks[home_team]
        lat, lon = park["lat"], park["lon"]
        stadium = park["stadium"]

        if not game_date:
            game_date = datetime.now().strftime('%Y-%m-%d')

        url = f"http://api.weatherapi.com/v1/forecast.json?key={self.api_key}&q={lat},{lon}&dt={game_date}"
        response = requests.get(url)
        data = response.json()

        if "forecast" not in data:
            raise Exception(f"Weather API error: {data.get('error', {}).get('message')}")

        hour_data = data["forecast"]["forecastday"][0]["hour"][game_hour]

        return {
            "stadium": stadium,
            "location": f"{lat},{lon}",
            "datetime": f"{game_date} {game_hour}:00",
            "condition": hour_data['condition']['text'],
            "temp_f": hour_data['temp_f'],
            "feelslike_f": hour_data['feelslike_f'],
            "humidity": hour_data['humidity'],
            "wind_mph": hour_data['wind_mph'],
            "wind_dir": hour_data['wind_dir'],
            "gust_mph": hour_data['gust_mph'],
            "precip_in": hour_data['precip_in'],
            "cloud": hour_data['cloud'],
            "uv": hour_data['uv']
        }

    def calculate_hr_multiplier(self, weather):
        multiplier = 1.00

        temp = weather["temp_f"]
        humidity = weather["humidity"]
        wind = weather["wind_mph"]
        wind_dir = weather["wind_dir"].upper()

        # Temperature boost (>70°F)
        if temp > 70:
            multiplier += 0.01 * ((temp - 70) // 5)

        # Humidity boost (>50%)
        if humidity > 50:
            multiplier += 0.01 * ((humidity - 50) // 10)

        # Wind adjustment
        if "OUT" in wind_dir or wind_dir in ("W", "WSW", "SW", "WNW"):
            multiplier += min(0.15, wind * 0.01)
        elif wind_dir in ("IN", "N", "NNE", "NNW"):
            multiplier -= min(0.10, wind * 0.01)

        return round(multiplier, 3)

    def pretty_print(self, weather, hr_mult=None):
        print(f"\n🌤️ Weather Forecast for {weather['stadium']} ({weather['datetime']}):")
        print(f"Condition: {weather['condition']}")
        print(f"Temperature: {weather['temp_f']}°F (Feels like: {weather['feelslike_f']}°F)")
        print(f"Humidity: {weather['humidity']}%")
        print(f"Wind: {weather['wind_mph']} mph {weather['wind_dir']}")
        print(f"Gusts: {weather['gust_mph']} mph")
        print(f"Precipitation: {weather['precip_in']} in")
        print(f"Cloud Cover: {weather['cloud']}%")
        print(f"UV Index: {weather['uv']}")
        if hr_mult:
            print(f"💣 Home Run Multiplier: x{hr_mult}")
