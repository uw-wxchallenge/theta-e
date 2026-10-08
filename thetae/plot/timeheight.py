#
# Copyright (c) 2019 Jonathan Weyn + Joe Zagrodnik <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Generates bufkit timeheight plots for various models.
"""

import os
from collections import OrderedDict
import re
import numpy as np
from scipy import interpolate
import pandas as pd
from math import ceil, floor
from datetime import datetime, timedelta
from io import open
import requests
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot_timeheight(config, stid, model, forecast_date, variable, df, plot_dir, img_type, run_time=None,
                    fallback=False, vrange_fixed=None):
    """
    Timeseries plotting function

    Currently designed to convert the Profile to a Pandas MultiIndex which is easy to manipulate for plotting.
    Future implementations should probably get rid of the Profile OrderedDict construct altogether.
    """

    # vertical pressure levels (y-coordinates)
    p_levels = np.flip(np.unique(df.index.get_level_values('pressure').values), 0)

    # slice dataframe depending on which variable we are plotting
    # select plotting variables, exclude models that do not have certain variables
    idx = pd.IndexSlice
    if variable == 'temperature':
        temperature = df.loc[idx[:, 'TMPC'], :]
        cmap = 'jet'
        vrange = [-20, 25]
        ytop = 650
        plot_variable = np.ma.masked_array(temperature.values.astype('float'))
        extend = 'both'
    elif variable == 'dewPointDep':
        dewpoint_dep = df.loc[idx[:, 'TMPC'], :].values - df.loc[idx[:, 'DWPC'], :].values
        cmap = 'BrBG_r'
        vrange = [0, 10]
        ytop = 200
        extend = 'max'
        plot_variable = np.ma.masked_array(dewpoint_dep.astype('float'))
    elif variable == 'cloud' and model[0:3] != 'GFS' and model[0:4] not in ['HRRR', 'RRFS']:
        cloud_fr = df.loc[idx[:, 'CFRL'], :]
        cmap = 'Blues_r'
        vrange = [0, 100]
        ytop = 200
        plot_variable = np.ma.masked_array(cloud_fr.values.astype('float'))
        extend = 'neither'
    elif variable == 'windSpeed':
        uwnd = df.loc[idx[:, 'UWND'], :].values.astype('float')
        vwnd = df.loc[idx[:, 'VWND'], :].values.astype('float')
        sknt = np.sqrt(uwnd ** 2 + vwnd ** 2)
        drct = np.ma.masked_array(180 / np.pi * np.arctan2(uwnd, vwnd))
        drct = np.ma.masked_where(np.isnan(drct), drct)
        drct += 180
        drct[np.where((drct < 0))] += 360
        cmap = 'jet'
        ytop = 650
        vrange = [0, int(ceil(np.nanmax(sknt[0:np.where((p_levels == ytop))[0][0], :] / 10.0))) * 10]
        extend = 'neither'
        plot_variable = sknt

        # select values for wind barbs
        barb_heights = [1000., 950., 900., 850., 800., 750., 700.]
        uwnd_barbs = np.zeros((len(barb_heights), len(uwnd[0, :])))
        vwnd_barbs = np.zeros((len(barb_heights), len(uwnd[0, :])))
        for i in range(0, len(barb_heights)):
            uwnd_barbs[i, :] = uwnd[np.where((p_levels == barb_heights[i])), :]
            vwnd_barbs[i, :] = vwnd[np.where((p_levels == barb_heights[i])), :]

        # get boundary-level winds
        bl_df = compute_bl_winds(df)
    elif variable == 'omega':
        omeg = df.loc[idx[:, 'OMEG'], :].values.astype('float')
        cmap = 'seismic'
        ytop = 200
        vrange = [-ceil(np.nanmax(np.abs(omeg) * 10)) / 10., ceil(np.nanmax(np.abs(omeg) * 10)) / 10.]
        extend = 'neither'
        plot_variable = omeg

        # also plot potential temperature contours on omega
        tmpk = df.loc[idx[:, 'TMPC'], :].values.astype('float') + 273.15
        pres = df.loc[idx[:, 'PRES'], :].values.astype('float')
        theta = tmpk * (1000 / pres) ** 0.286
    else:
        if config['debug'] > 50:
            print('%s Bufkit time-height data NOT processed--VARIABLE: %s for MODEL = %s' % (stid, variable, model))
        return

    if config['debug'] > 50:
        print('%s Bufkit time-height data processed--VARIABLE: %s for MODEL = %s' % (stid, variable, model))

    # Shared color scale across models (see common_vranges)
    if vrange_fixed is not None:
        vrange = vrange_fixed
        extend = 'neither'

    # model times (x-values)
    times = pd.to_datetime(df.loc[idx[:, 'TMPC'], :].columns)

    # set up the plot
    fig = plt.figure()
    fig.set_size_inches(8, 6)
    ax = fig.add_subplot(1, 1, 1)

    # get lower y limit by figuring out which pressure surfaces have data
    pres_surf = np.array(list(zip(*df.loc[idx[:, 'TMPC'], :].apply(pd.Series.first_valid_index).values.tolist()))[0])
    ylims = (np.max(pres_surf), ytop)

    # meshplot of variable
    plot_variable = np.ma.masked_where(np.isnan(plot_variable), plot_variable)
    meshplot = ax.pcolormesh(np.array(times), p_levels, plot_variable, cmap=cmap, vmin=vrange[0], vmax=vrange[1])

    # plot 0 deg C line on temperature plot
    if variable == 'temperature':
        zero_line = plt.contour(times, p_levels, plot_variable, [0.0], colors='k', linewidths=3,
                                linestyles='dashed')
        plt.clabel(zero_line, colors='k', inline_spacing=1, fmt='%1.0f C', rightside_up=True)
    # plot wind barbs and mixed-layer height on wind plot
    if variable == 'windSpeed':
        plt.barbs(matplotlib.dates.date2num(times.to_pydatetime()), barb_heights, uwnd_barbs, vwnd_barbs, length=5.5,
                  lw=0.5)
        bl_line = ax.plot(bl_df.index, bl_df['pressure'], color='red', linewidth=2, linestyle='dashed')
        plt.legend([bl_line[0]], ['Mixed Layer Height'], loc=2)
    if variable == 'omega':
        theta_contours = np.arange(0, 402, 2)
        theta_lines = plt.contour(times, p_levels, theta, theta_contours, colors='k', linewidths=1, linestyles='solid')
        plt.clabel(theta_lines, colors='k', inline_spacing=1, fmt='%d', rightside_up=True, fontsize=8)
        h1, _ = theta_lines.legend_elements()
        plt.legend([h1[0]], ['Potential Temperature (K)'], loc=2, framealpha=0.9)

    # vertical lines at start and end of forecast period
    ax.plot([forecast_date + timedelta(hours=6)] * 2, [p_levels[0], p_levels[-1]], color='k', lw=2.0)
    ax.plot([forecast_date + timedelta(hours=30)] * 2, [p_levels[0], p_levels[-1]], color='k', lw=2.0)
    # label end of model run if it is located on plot
    if times[-1] < (forecast_date + timedelta(hours=40)):
        ax.plot([times[-1]] * 2, [p_levels[0], p_levels[-1]], color='magenta', lw=2.0)
        ax.text(times[-1] + timedelta(hours=1), np.mean(ylims), '(end of model run)', color='magenta', rotation=90,
                ha='center', va='center')

    # Plot configurations and saving
    ax.grid()
    title = '{} forecast {} time-height at {}'.format(model, variable.upper(), stid)
    if run_time is not None:
        title += '\nrun {:%Y-%m-%d %HZ}'.format(run_time)
        if fallback:
            title += ' (latest run not yet available)'
    ax.set_title(title)
    ax.set_xlabel('Valid time')
    ax.set_ylabel('Pressure (hPa)')

    # axis range and label formatting
    from matplotlib import dates
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # An older (fallback) run starts a day earlier; keep the same window as a current run
    ax.set_xlim(max(times[0], forecast_date - timedelta(hours=12)), forecast_date + timedelta(hours=42))
    ax.set_ylim(ylims)
    ax.xaxis.set_major_locator(dates.HourLocator(byhour=list(range(0, 25, 3))))
    ax.xaxis.set_major_formatter(dates.DateFormatter('%HZ'))
    ax.xaxis.set_minor_locator(dates.DayLocator())
    ax.xaxis.set_minor_formatter(dates.DateFormatter('%h %d'))
    ax.xaxis.set_tick_params(which='minor', pad=15)

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)
    plt.colorbar(meshplot, cax, extend=extend)

    # Save plot
    plt.savefig('{}/{}_timeheight_{}_{}.{}'.format(plot_dir, stid, variable.upper(), model, img_type), dpi=150)
    plt.close()
    return


def timeheight_only_models(config):
    """
    Sounding models listed under [Plot][[TimeHeight]] (e.g. RAP, for its cloud fraction). They are plotted here
    only; they are not in [Models], so they never reach the forecast database, stats or other pages.
    """
    try:
        section = config['Plot']['TimeHeight']
    except KeyError:
        return OrderedDict()
    return OrderedDict((m, section[m]) for m in section.sections)


def sounding_model_config(config, model):
    if model in config['Models']:
        return config['Models'][model]
    return timeheight_only_models(config)[model]


def get_timeheight_only_bufkit(config, model, stid, forecast_date):
    """
    Download a time-height-only model's bufkit file through BUFRgruven, as bufkit.py does for [Models] entries.
    """
    from thetae.data_parsers.bufkit import get_bufkit_forecast
    model_config = timeheight_only_models(config)[model]
    bufkit_dir = config['BUFKIT']['BUFKIT_directory']
    file_name = '%s/bufkit/%s%s.%s_%s.buf' % (bufkit_dir, (forecast_date - timedelta(days=1)).strftime('%Y%m%d'),
                                              model_config['run_time'].replace('Z', ''), model_config['bufr_name'],
                                              stid.lower())
    if os.path.isfile(file_name):
        return
    try:
        get_bufkit_forecast(config, config['BUFKIT']['BUFR'], bufkit_dir, model, model_config['bufr_name'],
                            model_config['run_time'], stid, forecast_date)
    except BaseException as e:
        # Not posted yet; the plot falls back to the previous run
        if config['debug'] > 9:
            print('plot.timeheight: could not get bufkit file for %s: %s' % (model, e))


# Open-Meteo pressure levels used for time-height profiles (not every model has all of them)
openmeteo_levels = [1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200]
openmeteo_profile_vars = ['temperature', 'dew_point', 'cloud_cover', 'wind_speed', 'wind_direction',
                          'geopotential_height', 'vertical_velocity']


def openmeteo_timeheight_profile(config, model, stid, forecast_date):
    """
    Build a time-height DataFrame, in the same layout as bufr_timeheight_parser, from Open-Meteo pressure-level
    forecasts. Uses the model's 'profile_om_model' if set (ECMWF IFS 9 km has no pressure levels on Open-Meteo,
    so ECMWF uses the 0.25 degree IFS), else its 'om_model'. Values are interpolated in pressure to the same
    5 hPa levels as the soundings; levels below ground are dropped. Fields a model doesn't provide (e.g.
    vertical velocity for ICON and GEM) come back all-missing and are not plotted.
    """
    model_config = config['Models'][model]
    om_model = model_config.get('profile_om_model', model_config['om_model'])
    params = {
        'latitude': float(config['Stations'][stid]['latitude']),
        'longitude': float(config['Stations'][stid]['longitude']),
        'hourly': ','.join('%s_%dhPa' % (v, l) for v in openmeteo_profile_vars for l in openmeteo_levels),
        'models': om_model,
        'wind_speed_unit': 'kn',
        'timezone': 'GMT',
        'temporal_resolution': 'native',
        'start_date': (forecast_date - timedelta(days=1)).strftime('%Y-%m-%d'),
        'end_date': (forecast_date + timedelta(days=1)).strftime('%Y-%m-%d'),
    }
    response = requests.get('https://api.open-meteo.com/v1/forecast', params=params, timeout=120)
    if response.status_code == 400:
        raise IOError('plot.timeheight: Open-Meteo has no %s data for %s: %s' %
                      (om_model, stid, response.json().get('reason', '')))
    response.raise_for_status()
    data = response.json()
    hourly = data['hourly']
    elevation = data.get('elevation', 0.)
    times = pd.to_datetime(hourly['time'])
    keep = (times >= forecast_date - timedelta(hours=12)) & (times <= forecast_date + timedelta(hours=42))

    def level_array(var):
        values = [hourly.get('%s_%dhPa' % (var, l), [None] * len(times)) for l in openmeteo_levels]
        return np.array(values, dtype='float')[:, keep]  # (level, time)

    temperature = level_array('temperature')
    height = level_array('geopotential_height')
    fields = {
        'TMPC': temperature,
        'DWPC': level_array('dew_point'),
        'CFRL': level_array('cloud_cover'),
        'HGHT': height,
    }
    speed = level_array('wind_speed')
    direction = level_array('wind_direction')
    # Same u/v convention as the bufkit parser (knots)
    fields['UWND'] = speed * np.sin(direction * np.pi / 180. - np.pi)
    fields['VWND'] = speed * np.cos(direction * np.pi / 180. - np.pi)
    # Vertical velocity w (m/s) to omega in microbar/s, as in bufkit: omega = -rho g w, rho = p / (Rd T)
    w = level_array('vertical_velocity')
    pres_pa = np.array(openmeteo_levels, dtype='float')[:, None] * 100.
    rho = pres_pa / (287.05 * (temperature + 273.15))
    fields['OMEG'] = -rho * 9.81 * w * 10.

    # Interpolate each time's profile in pressure to the bufkit levels (1045 to 200 hPa every 5 hPa)
    plevs = list(range(200, 1050, 5))
    plevs.reverse()
    om_levels = np.array(openmeteo_levels, dtype='float')
    profile = OrderedDict()
    for t, valid_time in enumerate(times[keep]):
        above_ground = np.isfinite(height[:, t]) & (height[:, t] >= elevation)
        final_vars = OrderedDict()
        for var, values in fields.items():
            column = values[:, t]
            ok = above_ground & np.isfinite(column)
            if ok.sum() >= 2:
                f = interpolate.interp1d(om_levels[ok], column[ok], bounds_error=False)
                final_vars[var] = list(f(plevs))
            else:
                final_vars[var] = [np.nan] * len(plevs)
        final_vars['PRES'] = plevs
        profile[valid_time.to_pydatetime()] = final_vars
    if not profile:
        raise IOError('plot.timeheight: no Open-Meteo profile times for %s' % model)

    variables = list(profile[list(profile.keys())[0]].keys())
    index = pd.MultiIndex.from_tuples(list(zip(plevs * len(variables), np.repeat(variables, len(plevs)))),
                                      names=['pressure', 'var'])
    df = pd.DataFrame(index=index, columns=list(profile.keys()))
    for var in variables:
        for dt in profile.keys():
            df[dt].loc[:, var] = np.array(profile[dt][var])
    return df


def bufr_timeheight_parser(config, model, stid, forecast_date, model_date=None):
    """
    Original code by Luke Madaus, modified by Joe Zagrodnik and Jonathan Weyn

    Grab all variables, put in pandas dataframes

    """
    # Load bufkit file
    bufkit_dir = config['BUFKIT']['BUFKIT_directory']
    model_run_hour = sounding_model_config(config, model)['run_time'].replace('Z', '')
    bufr_name = sounding_model_config(config, model)['bufr_name']
    if model_date is None:
        model_date = forecast_date - timedelta(days=1)
    model_date = model_date.strftime('%Y%m%d')
    file_name = '%s/bufkit/%s%s.%s_%s.buf' % (bufkit_dir, model_date, model_run_hour, bufr_name, stid.lower())
    try:
        infile = open(file_name, 'r')
    except IOError:
        raise IOError('plot.timeheight: missing bufkit file %s' % file_name)

    profile = OrderedDict()

    # Find the block that contains the description of
    # what everything is (header information)
    block_lines = []
    inblock = False
    block_found = False
    for line in infile:
        if line.startswith('PRES TMPC') and not block_found:
            # We've found the line that starts the header info
            inblock = True
            block_lines.append(line)
        elif inblock:
            # Keep appending lines until we start hitting numbers
            if re.match(r'^\d{3}|^\d{4}', line):
                inblock = False
                block_found = True
            else:
                block_lines.append(''.join(line.split('\r')))  # jweyn: remove any returns

    # Now compute the remaining number of variables
    re_string = ''
    for line in block_lines:
        dum_num = len(line.split())
        for n in range(dum_num):
            re_string = re_string + r'(-?\d{1,5}.\d{2}) '
        re_string = re_string[:-1]  # Get rid of the trailing space
        re_string = re_string + r'\n'

    # Compile this re_string for more efficient re searches
    block_expr = re.compile(re_string)

    # Now get corresponding indices of the variables we need
    full_line = ''
    for r in block_lines:
        full_line = full_line + ''.join(r.split('\n')) + ' '
    # Now split it
    varlist = full_line.strip().split(' ')

    # Variables we want
    vars_desired = ['TMPC', 'DWPC', 'UWND', 'VWND', 'HGHT', 'OMEG', 'CFRL']

    # Pressure levels to interpolate to
    interp_res = 5
    plevs = range(200, 1050, interp_res)

    # We now need to break everything up into a chunk for each
    # forecast date and time
    infile.seek(0)
    blocks = infile.read().split('STID')
    infile.close()
    for block in blocks:
        interp_plevs = []
        header = block
        if header.split()[0] != '=':
            continue
        fcst_time = re.search(r'TIME = (\d{6}/\d{4})', header).groups()[0]
        fcst_dt = datetime.strptime(fcst_time, '%y%m%d/%H%M')

        # End loop if we are more than 60 hours past the start of the forecast date
        if fcst_dt > forecast_date + timedelta(hours=60):
            break
        temp_vars = OrderedDict()
        for var in varlist:
            temp_vars[var] = []
        temp_vars['PRES'] = []
        for block_match in block_expr.finditer(block):
            vals = block_match.groups()
            for val, name in zip(vals, varlist):
                if float(val) == -9999.:
                    temp_vars[name].append(np.nan)
                else:
                    temp_vars[name].append(float(val))

        # Unfortunately, bufkit values aren't always uniformly distributed.
        final_vars = OrderedDict()
        cur_plevs = temp_vars['PRES']
        cur_plevs.reverse()
        for var in varlist[1:]:
            if var in (vars_desired + ['SKNT', 'DRCT']):
                values = temp_vars[var]
                values.reverse()
                interp_plevs = list(plevs)
                num_plevs = len(interp_plevs)
                f = interpolate.interp1d(cur_plevs, values, bounds_error=False)
                interp_vals = f(interp_plevs)
                interp_array = np.full((len(plevs)), np.nan)
                # Array almost certainly missing values at high pressures
                interp_array[:num_plevs] = interp_vals
                interp_vals = list(interp_array)
                interp_plevs = list(plevs)  # use original array
                interp_vals.reverse()
                interp_plevs.reverse()
                if var == 'SKNT':
                    wspd = np.array(interp_vals)
                if var == 'DRCT':
                    wdir = np.array(interp_vals)
            if var in vars_desired:
                final_vars[var] = interp_vals
        final_vars['PRES'] = interp_plevs
        if 'UWND' not in final_vars.keys():
            final_vars['UWND'] = list(wspd * np.sin(wdir * np.pi / 180. - np.pi))
        if 'VWND' not in final_vars.keys():
            final_vars['VWND'] = list(wspd * np.cos(wdir * np.pi / 180. - np.pi))
        profile[fcst_dt] = final_vars

    # convert Profile (OrderedDict) into MultiIndex DataFrame
    bufr_vars = list(profile[list(profile.keys())[0]].keys())
    pressure_inds = profile[list(profile.keys())[0]]['PRES'] * len(bufr_vars)
    bufr_vars_inds = np.repeat(bufr_vars, len(profile[list(profile.keys())[0]]['PRES']))
    index = pd.MultiIndex.from_tuples(list(zip(pressure_inds, bufr_vars_inds)), names=['pressure', 'var'])
    df = pd.DataFrame(index=index, columns=list(profile.keys()))

    # populate DataFrame from Profile
    for var in bufr_vars:
        for dt in profile.keys():
            df[dt].loc[:, var] = np.array(profile[dt][var])
    return df


def compute_bl_winds(bufkit_df):
    """
    Computes mixed-layer height and mixed-layer winds from bufkit profiles
    Works by searching for the height where the lapse rate is no longer close to dry adiabatic
    Function written by Luke Madaus, modified by Joe Zagrodnik

    :param bufkit_df: dataframe generated by bufr_timeheight_parser function
    :return: dataframe of mixed-layer height, mean and max mixed-layer wind

    """
    idx = pd.IndexSlice
    tmpc = bufkit_df.loc[idx[:, 'TMPC'], :].values.astype('float')
    hght = bufkit_df.loc[idx[:, 'HGHT'], :].values.astype('float')
    uwnd = bufkit_df.loc[idx[:, 'UWND'], :].values.astype('float')
    vwnd = bufkit_df.loc[idx[:, 'VWND'], :].values.astype('float')
    sknt = np.sqrt(uwnd ** 2 + vwnd ** 2)
    times = pd.to_datetime(bufkit_df.loc[idx[:, 'TMPC'], :].columns)
    p_levels = np.flip(np.unique(bufkit_df.index.get_level_values('pressure').values), 0)

    # Compute the gradients of height and temperature
    dh = np.gradient(hght, axis=0)
    dt = np.gradient(tmpc, axis=0)
    dh_km = dh / 1000.
    dtdz = np.divide(dt, dh_km)

    # dataframe to store boundary layer wind values
    bl_df = pd.DataFrame(index=times, columns=['height', 'pressure', 'mean_wind', 'max_wind'])

    for i in range(0, len(dtdz[0, :])):
        # free atmosphere starts at first level where dT/dZ > -6.0 degC/km
        free_atm = np.where((dtdz[:, i] > -6.0))[0]
        b_loc = np.min(free_atm)
        bl_df['height'][i] = hght[b_loc, i]
        bl_df['pressure'][i] = p_levels[b_loc]
        bl_df['mean_wind'][i] = np.nanmean(sknt[0:b_loc, i])
        bl_df['max_wind'][i] = np.nanmax(sknt[0:b_loc, i])

    return bl_df


# Top of the plotted pressure range for each variable (matches plot_timeheight)
plot_tops = {'temperature': 650, 'dewPointDep': 200, 'windSpeed': 650, 'omega': 200}


def plotted_values(df, variable, forecast_date):
    """
    Values of a time-height variable inside the plotted window (surface to the plot top, and the plotted times).
    Returns None for variables without a shared scale.
    """
    idx = pd.IndexSlice
    if variable not in plot_tops:
        return None
    tmpc = df.loc[idx[:, 'TMPC'], :]
    if variable == 'temperature':
        values = tmpc.values.astype('float')
    elif variable == 'dewPointDep':
        values = tmpc.values.astype('float') - df.loc[idx[:, 'DWPC'], :].values.astype('float')
    elif variable == 'windSpeed':
        uwnd = df.loc[idx[:, 'UWND'], :].values.astype('float')
        vwnd = df.loc[idx[:, 'VWND'], :].values.astype('float')
        values = np.sqrt(uwnd ** 2 + vwnd ** 2)
    else:
        values = df.loc[idx[:, 'OMEG'], :].values.astype('float')
    pressure = tmpc.index.get_level_values('pressure').values.astype('float')
    times = pd.to_datetime(tmpc.columns)
    rows = pressure >= plot_tops[variable]
    cols = (times >= max(times[0], forecast_date - timedelta(hours=12))) & \
           (times <= forecast_date + timedelta(hours=42))
    return values[np.ix_(rows, cols)]


def common_vranges(model_data, forecast_date, variables):
    """
    Color-scale limits per variable that bound the lowest and highest plotted value across all models, so the
    same color means the same value on every model's plot. Omega stays centered on zero.
    """
    vranges = {}
    for variable in variables:
        values = [plotted_values(d[0], variable, forecast_date) for d in model_data.values()]
        values = [v[np.isfinite(v)] for v in values if v is not None]
        values = [v for v in values if v.size > 0]
        if not values:
            continue
        vmin = min(np.min(v) for v in values)
        vmax = max(np.max(v) for v in values)
        if variable == 'omega':
            bound = ceil(max(abs(vmin), abs(vmax)) * 10) / 10.
            vranges[variable] = [-bound, bound]
        elif variable == 'windSpeed':
            vranges[variable] = [floor(vmin / 5.) * 5, ceil(vmax / 5.) * 5]
        else:
            vranges[variable] = [floor(vmin), ceil(vmax)]
    return vranges


def delete_plots(stid, model, variables, plot_dir, img_type):
    for v in variables:
        plot_file = '{}/{}_timeheight_{}_{}.{}'.format(plot_dir, stid, v.upper(), model, img_type)
        try:
            os.remove(plot_file)
        except:
            pass


def main(config, stid, forecast_date):
    """
    Make timeseries plots for a given station.
    """
    # Use the previous date if we're not at 6Z yet
    if datetime.utcnow().hour < 6:
        forecast_date -= timedelta(days=1)

    # Get the file directory and attempt to create it if it doesn't exist
    try:
        plot_directory = config['Plot']['Options']['plot_directory']
    except KeyError:
        plot_directory = '%s/site_data' % config['THETAE_ROOT']
        print('plot.timeheight warning: setting output directory to default')
    if not (os.path.isdir(plot_directory)):
        os.makedirs(plot_directory)
    if config['debug'] > 9:
        print('plot.timeheight: writing output to %s' % plot_directory)
    try:
        image_type = config['Plot']['Options']['plot_file_format']
    except KeyError:
        image_type = 'svg'
        if config['debug'] > 50:
            print('plot.timeheight warning: using default image file format (svg)')

    # Sounding models: [Models] entries with a bufkit file, then time-height-only models such as RAP
    sounding_models = [m for m in config['Models'].keys() if 'bufr_name' in config['Models'][m].keys()]
    extra_models = list(timeheight_only_models(config).keys())
    for model in extra_models:
        get_timeheight_only_bufkit(config, model, stid, forecast_date)
    # Open-Meteo models get profiles from Open-Meteo pressure-level forecasts
    openmeteo_models = [m for m in config['Models'].keys()
                        if config['Models'][m].get('driver') == 'thetae.data_parsers.openmeteo']

    # List of time-height variables
    variables = ['temperature', 'dewPointDep', 'cloud', 'windSpeed', 'omega']

    # Read every model first, so all plots can share one color scale per variable
    model_data = OrderedDict()
    for model in sounding_models + extra_models:
        # Use the current run (from the day before forecast_date). If it isn't posted yet (12Z runs
        # arrive around 16Z, 18Z runs around 22Z), plot the previous day's run instead of leaving
        # no plot. This relies on [BUFKIT] archive = True keeping older bufkit files.
        run_hour = int(sounding_model_config(config, model)['run_time'].replace('Z', ''))
        df = None
        parse_error = False
        for days_back, fallback in [(1, False), (2, True)]:
            model_date = forecast_date - timedelta(days=days_back)
            try:
                df = bufr_timeheight_parser(config, model, stid, forecast_date, model_date=model_date)
                break
            except IOError:
                continue
            except BaseException:
                if config['traceback']:
                    raise
                parse_error = True
                break
        if parse_error:
            continue
        if df is None:
            if config['debug'] > 9:
                print('plot.timeheight: bufr file missing for %s; deleting old plots' % model)
            delete_plots(stid, model, variables, plot_directory, image_type)
            continue
        model_data[model] = (df, model_date.replace(hour=run_hour), fallback)

    for model in openmeteo_models:
        try:
            df = openmeteo_timeheight_profile(config, model, stid, forecast_date)
        except BaseException as e:
            if config['debug'] > 9:
                print('plot.timeheight: no Open-Meteo profile for %s (%s); deleting old plots' % (model, e))
            delete_plots(stid, model, variables, plot_directory, image_type)
            continue
        model_data[model] = (df, None, False)

    try:
        vranges = common_vranges(model_data, forecast_date, variables)
    except BaseException:
        if config['traceback']:
            raise
        vranges = {}

    for model, (df, run_time, fallback) in model_data.items():
        for variable in variables:
            # Skip fields this model doesn't provide (e.g. omega for ICON/GEM from Open-Meteo, cloud for
            # GFS/HRRR/RRFS soundings) rather than writing an empty plot
            try:
                values = plotted_values(df, variable, forecast_date) if variable != 'cloud' else \
                    df.loc[pd.IndexSlice[:, 'CFRL'], :].values.astype('float')
            except KeyError:
                values = np.array([np.nan])
            if values is not None and not np.isfinite(values).any():
                delete_plots(stid, model, [variable], plot_directory, image_type)
                continue
            if config['debug'] > 50:
                print('plot.timeheight: plotting %s for %s' % (variable, model))
            try:
                plot_timeheight(config, stid, model, forecast_date, variable, df, plot_directory, image_type,
                                run_time=run_time, fallback=fallback, vrange_fixed=vranges.get(variable))
            except BaseException:
                if config['traceback']:
                    raise
    return
