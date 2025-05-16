import requests
from datetime import datetime
import numpy as np

class BallparkWeather:
    def __init__(self, api_key):
        self.api_key = api_key
        self.weather_cache = {}
        self.team_ballparks = {
            "ARI": {"stadium": "Chase Field", "lat": 33.4455, "lon": -112.0667, "elevation_ft": 1086},
            "ATL": {"stadium": "Truist Park", "lat": 33.8908, "lon": -84.4678, "elevation_ft": 1000},
            "BAL": {"stadium": "Camden Yards", "lat": 39.2839, "lon": -76.6218, "elevation_ft": 33},
            "BOS": {"stadium": "Fenway Park", "lat": 42.3467, "lon": -71.0972, "elevation_ft": 21},
            "CHC": {"stadium": "Wrigley Field", "lat": 41.9484, "lon": -87.6553, "elevation_ft": 600},
            "CIN": {"stadium": "Great American Ball Park", "lat": 39.0972, "lon": -84.5072, "elevation_ft": 492},
            "CLE": {"stadium": "Progressive Field", "lat": 41.4962, "lon": -81.6852, "elevation_ft": 653},
            "COL": {"stadium": "Coors Field", "lat": 39.7559, "lon": -104.9942, "elevation_ft": 5197},
            "CWS": {"stadium": "Guaranteed Rate Field", "lat": 41.8299, "lon": -87.6339, "elevation_ft": 595},
            "DET": {"stadium": "Comerica Park", "lat": 42.3390, "lon": -83.0490, "elevation_ft": 602},
            "HOU": {"stadium": "Minute Maid Park", "lat": 29.7572, "lon": -95.3556, "elevation_ft": 46},
            "KC": {"stadium": "Kauffman Stadium", "lat": 39.0516, "lon": -94.4804, "elevation_ft": 886},
            "LAA": {"stadium": "Angel Stadium", "lat": 33.8003, "lon": -117.8827, "elevation_ft": 154},
            "LAD": {"stadium": "Dodger Stadium", "lat": 34.0739, "lon": -118.2400, "elevation_ft": 501},
            "MIA": {"stadium": "LoanDepot Park", "lat": 25.7780, "lon": -80.2195, "elevation_ft": 7},
            "MIL": {"stadium": "American Family Field", "lat": 43.0280, "lon": -87.9711, "elevation_ft": 602},
            "MIN": {"stadium": "Target Field", "lat": 44.9817, "lon": -93.2789, "elevation_ft": 839},
            "NYM": {"stadium": "Citi Field", "lat": 40.7571, "lon": -73.8458, "elevation_ft": 13},
            "NYY": {"stadium": "Yankee Stadium", "lat": 40.8296, "lon": -73.9262, "elevation_ft": 52},
            "OAK": {"stadium": "Oakland Coliseum", "lat": 37.7516, "lon": -122.2005, "elevation_ft": 6},
            "PHI": {"stadium": "Citizens Bank Park", "lat": 39.9061, "lon": -75.1665, "elevation_ft": 20},
            "PIT": {"stadium": "PNC Park", "lat": 40.4469, "lon": -80.0057, "elevation_ft": 724},
            "SD": {"stadium": "Petco Park", "lat": 32.7073, "lon": -117.1573, "elevation_ft": 20},
            "SEA": {"stadium": "T-Mobile Park", "lat": 47.5914, "lon": -122.3325, "elevation_ft": 15},
            "SF": {"stadium": "Oracle Park", "lat": 37.7786, "lon": -122.3893, "elevation_ft": 0},
            "STL": {"stadium": "Busch Stadium", "lat": 38.6226, "lon": -90.1928, "elevation_ft": 465},
            "TB": {"stadium": "Tropicana Field", "lat": 27.7683, "lon": -82.6534, "elevation_ft": 15},
            "TEX": {"stadium": "Globe Life Field", "lat": 32.7473, "lon": -97.0847, "elevation_ft": 551},
            "TOR": {"stadium": "Rogers Centre", "lat": 43.6414, "lon": -79.3894, "elevation_ft": 253},
            "WSH": {"stadium": "Nationals Park", "lat": 38.8728, "lon": -77.0075, "elevation_ft": 7},
        }
        # Precomputed regression coefficients (assumed from historical data)
        self.weather_coefficients = {
            'temp_f': 0.002,  # Per degree Fahrenheit
            'humidity': 0.001,  # Per percentage point
            'wind_mph': 0.005,  # Per mph
            'wind_favorable': 0.1,  # Binary for favorable wind direction
            'air_density': -0.05,  # Per kg/m^3
            'intercept': 1.0  # Baseline multiplier
        }

    def get_home_team(self, matchup_str):
        parts = matchup_str.strip().upper().split("VS")
        if len(parts) != 2:
            raise ValueError("Matchup format must be like 'NYY vs SEA'")
        return parts[1].strip()

    def calculate_air_density(self, weather, elevation_ft):
        """Calculate air density in kg/m^3 based on temperature, humidity, and elevation."""
        temp_k = (weather['temp_f'] - 32) * 5/9 + 273.15  # Convert to Kelvin
        pressure = 1013.25 * (1 - 2.25577e-5 * elevation_ft)**5.25588  # Pressure in hPa
        rh = weather['humidity'] / 100
        vapor_pressure = rh * 6.1078 * 10**((7.5 * (temp_k - 273.15)) / (temp_k - 35.85))
        air_density = (pressure * 100 / (287.05 * temp_k)) * (1 - 0.378 * vapor_pressure / pressure)
        return air_density

    def get_forecast(self, matchup_str, game_hour=19, game_date=None):
        home_team = self.get_home_team(matchup_str)
        if home_team not in self.team_ballparks:
            raise ValueError(f"Unknown team code: {home_team}")

        park = self.team_ballparks[home_team]
        lat, lon = park["lat"], park["lon"]
        stadium = park["stadium"]

        # Validate game_date
        if not game_date:
            game_date = datetime.now().strftime('%Y-%m-%d')
        try:
            datetime.strptime(game_date, '%Y-%m-%d')
        except ValueError:
            print(f"Couldn't get weather data for {matchup_str} at {game_date} {game_hour}:00: Invalid date format")
            return None

        # Validate game_hour
        if not isinstance(game_hour, int) or game_hour < 0 or game_hour > 23:
            print(f"Couldn't get weather data for {matchup_str} at {game_date} {game_hour}:00: Invalid hour")
            return None

        cache_key = f"{matchup_str}_{game_date}_{game_hour}"
        if cache_key in self.weather_cache:
            return self.weather_cache[cache_key]

        url = f"http://api.weatherapi.com/v1/forecast.json?key={self.api_key}&q={lat},{lon}&dt={game_date}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200 or not response.text:
                print(f"Couldn't get weather data for {matchup_str} at {game_date} {game_hour}:00")
                return None

            data = response.json()
            if "error" in data:
                print(f"Couldn't get weather data for {matchup_str} at {game_date} {game_hour}:00: {data['error']['message']}")
                return None
            if not data.get("forecast") or not data["forecast"].get("forecastday") or not data["forecast"]["forecastday"]:
                print(f"Couldn't get weather data for {matchup_str} at {game_date} {game_hour}:00")
                return None

            hour_data = data["forecast"]["forecastday"][0]["hour"][game_hour]

            required_fields = ['condition', 'temp_f', 'feelslike_f', 'humidity', 'wind_mph', 'wind_dir', 'gust_mph', 'precip_in', 'cloud', 'uv']
            if not all(field in hour_data and isinstance(hour_data['condition'], dict) and 'text' in hour_data['condition'] for field in required_fields):
                print(f"Couldn't get weather data for {matchup_str} at {game_date} {game_hour}:00: Incomplete forecast data")
                return None

            forecast = {
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
            self.weather_cache[cache_key] = forecast
            return forecast
        except (requests.RequestException, ValueError, IndexError, KeyError) as e:
            print(f"Couldn't get weather data for {matchup_str} at {game_date} {game_hour}:00")
            return None

    def calculate_hr_multiplier(self, weather, home_team):
        """Calculate home run probability multiplier using a data-driven regression model."""
        if not weather:
            return 1.00
        
        # Get elevation for air density
        elevation_ft = self.team_ballparks.get(home_team, {}).get('elevation_ft', 0)
        
        # Calculate air density
        air_density = self.calculate_air_density(weather, elevation_ft)
        
        # Encode wind direction (favorable: W, WSW, SW, WNW, or "OUT")
        wind_dir = weather['wind_dir'].upper()
        wind_favorable = 1 if "OUT" in wind_dir or wind_dir in ("W", "WSW", "SW", "WNW") else 0
        
        # Apply regression model
        features = {
            'temp_f': weather['temp_f'],
            'humidity': weather['humidity'],
            'wind_mph': weather['wind_mph'],
            'wind_favorable': wind_favorable,
            'air_density': air_density
        }
        
        multiplier = self.weather_coefficients['intercept']
        for feature, value in features.items():
            multiplier += self.weather_coefficients[feature] * value
        
        # Ensure multiplier is reasonable (e.g., between 0.8 and 1.2)
        return round(max(0.8, min(1.2, multiplier)), 3)

    def pretty_print(self, weather, hr_mult=None, park_factor=None):
        if not weather:
            print("No weather data available")
            return
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
            print(f"💣 Weather Home Run Multiplier: x{hr_mult}")
        if park_factor:
            print(f"🏟️ Park Factor (HR): {park_factor}")