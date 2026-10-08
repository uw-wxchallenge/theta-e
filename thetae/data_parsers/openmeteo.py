#
# Copyright (c) 2026 UW WxChallenge team
#
# See the file LICENSE for your rights.
#

"""
Retrieve deterministic model forecasts from the Open-Meteo forecast API (https://open-meteo.com).
One driver serves many models; the conf option 'om_model' is Open-Meteo's models= value, e.g.
ecmwf_ifs, icon_global, gem_regional, gem_hrdps_continental, ukmo_global_deterministic_10km.

Open-Meteo is free for non-commercial use with no API key. The data are CC-BY 4.0 licensed, so
the site must credit Open-Meteo.
"""

from thetae import Forecast
from thetae.util import wind_day_window
from datetime import timedelta
import requests
import pandas as pd
import numpy as np

default_model_name = 'OpenMeteo'

api_url = 'https://api.open-meteo.com/v1/forecast'

# Open-Meteo hourly variable name -> HOURLY_FORECAST column name
variables = {
    'temperature_2m': 'temperature',
    'dew_point_2m': 'dewpoint',
    'cloud_cover': 'cloud',
    'wind_speed_10m': 'windSpeed',
    'wind_gusts_10m': 'windGust',
    'wind_direction_10m': 'windDirection',
    'precipitation': 'rain',
    'pressure_msl': 'pressure',
}


def get_openmeteo_forecast(stid, model, om_model, lat, lon, forecast_date, wind_window):
    """
    Retrieve an Open-Meteo forecast for one model at a lat/lon.

    :param stid: str: station ID
    :param model: str: theta-e model name
    :param om_model: str: Open-Meteo models= value
    :param lat: float: latitude
    :param lon: float: longitude
    :param forecast_date: datetime: forecast day (00Z)
    :return: Forecast
    """
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': ','.join(variables.keys()),
        'models': om_model,
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit': 'kn',
        'precipitation_unit': 'inch',
        'timezone': 'GMT',
        'start_date': forecast_date.strftime('%Y-%m-%d'),
        'end_date': (forecast_date + timedelta(days=1)).strftime('%Y-%m-%d'),
    }
    response = requests.get(api_url, params=params, timeout=60)

    # A point outside a regional model's domain (e.g. HRDPS south of Canada) returns HTTP 400 with a reason
    if response.status_code == 400:
        try:
            reason = response.json().get('reason', '')
        except ValueError:
            reason = ''
        if 'No data is available for this location' in reason:
            raise ValueError('openmeteo: %s (%s) has no data for %s (outside model domain)' % (model, om_model, stid))
        print('openmeteo: got HTTP 400 for %s: %s' % (model, reason))
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        print('openmeteo: got HTTP error when querying API for %s' % model)
        raise

    hourly_data = response.json()['hourly']

    # Create a DataFrame for hourly data, indexed by naive UTC time
    hourly = pd.DataFrame()
    hourly['DateTime'] = pd.to_datetime(hourly_data['time'])
    for om_var, column in variables.items():
        hourly[column] = pd.to_numeric(pd.Series(hourly_data.get(om_var, [None] * len(hourly))), errors='coerce')
    hourly['datetime_index'] = hourly['DateTime']
    hourly.set_index('datetime_index', inplace=True)

    # Aggregate daily values over the 06Z-06Z day. Open-Meteo precipitation is the sum over the
    # preceding hour, so the day's rain is the hours ending 07Z through 06Z. Any missing hour (e.g. the
    # run doesn't reach the end of the day) leaves that daily value as None rather than a partial value.
    forecast_start = forecast_date.replace(hour=6)
    forecast_end = forecast_start + timedelta(days=1)
    day = hourly.loc[forecast_start:forecast_end]
    day_rain = hourly.loc[forecast_start + timedelta(hours=1):forecast_end, 'rain']

    def daily_value(series, func, n_hours):
        if len(series) < n_hours or series.isnull().any():
            return None
        return func(series)

    high = daily_value(day['temperature'], lambda s: int(np.round(s.max())), 25)
    low = daily_value(day['temperature'], lambda s: int(np.round(s.min())), 25)
    # Max wind is over the local midnight-to-midnight day (as in the NWS climate report), not 06Z-06Z
    wind_hours = int((wind_window[1] - wind_window[0]).total_seconds() // 3600)
    wind_day = hourly.loc[wind_window[0]:wind_window[1] - timedelta(seconds=1), 'windSpeed']
    wind = daily_value(wind_day, lambda s: int(np.round(s.max())), wind_hours)
    rain = daily_value(day_rain, lambda s: round(float(s.sum()), 3), 24)

    # Create the Forecast object
    forecast = Forecast(stid, model, forecast_date)
    forecast.daily.set_values(high, low, wind, rain)
    forecast.timeseries.data = hourly

    return forecast


def main(config, model, stid, forecast_date):
    """
    Produce a Forecast object from Open-Meteo.
    """
    try:
        om_model = config['Models'][model]['om_model']
    except KeyError:
        raise KeyError("openmeteo: no 'om_model' parameter defined for model %s in config!" % model)
    try:
        lat = float(config['Stations'][stid]['latitude'])
        lon = float(config['Stations'][stid]['longitude'])
    except (KeyError, ValueError):
        raise KeyError('openmeteo: missing or invalid latitude or longitude for station %s' % stid)

    forecast = get_openmeteo_forecast(stid, model, om_model, lat, lon, forecast_date,
                                      wind_day_window(config, stid, forecast_date))

    return forecast
