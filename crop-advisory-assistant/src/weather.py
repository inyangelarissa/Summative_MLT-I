"""Weather context for the advisory model.

The model was trained with a weather block: temperature (°C), relative humidity (%), rainfall over the last
7 days (mm) and a season label. This module produces the same four fields either from user input or from the
free Open-Meteo API (no key needed; needs internet at run time). Values are coarse regional summaries, not
field measurements: say so in the report.
"""
import json
import urllib.parse
import urllib.request

from build_climate import season_label

# Approximate district centres (lat, lon). Check/extend as needed.
RWANDA_LOCATIONS = {
    "Kigali": (-1.9441, 30.0619), "Musanze": (-1.4998, 29.6344), "Huye": (-2.5967, 29.7394),
    "Rubavu": (-1.7028, 29.2563), "Nyagatare": (-1.2975, 30.3270), "Rusizi": (-2.4844, 28.9070),
    "Muhanga": (-2.0850, 29.7560), "Kayonza": (-1.8836, 30.6339), "Nyamagabe": (-2.4700, 29.5700),
}


def climate_from_values(temperature_c, humidity_pct, rainfall_mm_7d):
    rain = int(round(rainfall_mm_7d))
    return {"temperature_c": int(round(temperature_c)), "humidity_pct": int(round(humidity_pct)),
            "rainfall_mm_7d": rain, "season": season_label(rain)}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize_hourly(hourly, hours=168):
    """hourly = Open-Meteo 'hourly' dict. Uses the first `hours` entries (the past 7 days when the request
    uses past_days=7&forecast_days=1). Returns the climate dict, or None if data is missing."""
    t = _mean(hourly.get("temperature_2m", [])[:hours])
    h = _mean(hourly.get("relative_humidity_2m", [])[:hours])
    precip = [x for x in hourly.get("precipitation", [])[:hours] if x is not None]
    if t is None or h is None or not precip:
        return None
    return climate_from_values(t, h, sum(precip))


def fetch_week_summary(lat, lon, timeout=10, opener=urllib.request.urlopen):
    q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "past_days": 7, "forecast_days": 1,
                                "hourly": "temperature_2m,relative_humidity_2m,precipitation",
                                "timezone": "Africa/Kigali"})
    with opener(f"https://api.open-meteo.com/v1/forecast?{q}", timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return summarize_hourly(data["hourly"])
