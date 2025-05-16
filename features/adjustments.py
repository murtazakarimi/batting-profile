import pandas as pd
from weather import BallparkWeather

def apply_weather_and_park_adjustments(batter_performance, weather, home_team, park_factors):
    """Apply weather and park factor adjustments to HR probabilities."""
    weather_adjustment = 1.0
    if weather:
        weather_obj = BallparkWeather(api_key=None)
        weather_adjustment = weather_obj.calculate_hr_multiplier(weather, home_team)
    
    park_factor_adjustment = 1.0
    if park_factors and home_team:
        park_factor = park_factors.get(home_team, 100)
        park_factor_adjustment = park_factor / 100
    
    batter_performance['weather_adjust'] = (weather_adjustment - 1) * 100
    batter_performance['park_factor_adjust'] = (park_factor_adjustment - 1) * 100
    batter_performance['weather_adjustment_factor'] = weather_adjustment
    batter_performance['park_factor_adjustment_factor'] = park_factor_adjustment
    
    return batter_performance