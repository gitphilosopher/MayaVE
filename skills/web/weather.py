"""
skills/web/weather.py
Fetches current weather + today's forecast from Open-Meteo (free, no API key).
Uses ip-api.com to auto-detect location if user doesn't specify one.
"""

import asyncio
import logging
import httpx

from config.settings import config

logger = logging.getLogger(__name__)
_U = config.user_name

_GEO_URL     = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
_IP_URL      = "http://ip-api.com/json/?fields=city,regionName,country,lat,lon"

_WMO = {
    0: "clear skies", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "icy fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "rain showers", 81: "showers", 82: "heavy showers",
    95: "thunderstorms", 96: "thunderstorms with hail",
}

# Map weather codes to expressions
_WMO_EXPRESSION = {
    0: "happy", 1: "happy", 2: "relaxed", 3: "neutral",
    45: "sad", 48: "sad",
    51: "sad", 53: "sad", 55: "sad",
    61: "sad", 63: "sad", 65: "angry",
    71: "surprised", 73: "surprised", 75: "angry",
    80: "sad", 81: "sad", 82: "angry",
    95: "angry", 96: "angry",
}


async def execute(intent: dict, text: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _fetch, text)


def _fetch(text: str) -> str:
    try:
        location = _extract_location(text)

        with httpx.Client(timeout=8.0) as client:
            if location:
                lat, lon, city = _geocode(client, location)
            else:
                lat, lon, city = _ip_location(client)

            params = {
                "latitude":  lat, "longitude": lon,
                "current":   "temperature_2m,weathercode,windspeed_10m,relativehumidity_2m",
                "daily":     "temperature_2m_max,temperature_2m_min,weathercode",
                "timezone":  "auto",
                "forecast_days": 1,
            }
            resp = client.get(_WEATHER_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        cur      = data["current"]
        code     = cur["weathercode"]
        temp     = round(cur["temperature_2m"])
        desc     = _WMO.get(code, "unknown conditions")
        wind     = round(cur["windspeed_10m"])
        hum      = cur["relativehumidity_2m"]
        expr     = _WMO_EXPRESSION.get(code, "neutral")

        d  = data["daily"]
        hi = round(d["temperature_2m_max"][0])
        lo = round(d["temperature_2m_min"][0])

        return (
            f"[{expr}] It's {temp}°C and {desc} in {city}, {_U}. "
            f"[relaxed] Wind at {wind} km/h, humidity {hum}%. "
            f"[neutral] Today's high is {hi}°C and low is {lo}°C."
        )

    except httpx.TimeoutException:
        return f"[sad] Weather service is taking too long, {_U}. Try again in a moment."
    except Exception as e:
        logger.error(f"Weather error: {e}", exc_info=True)
        return f"[sad] I couldn't fetch the weather right now, {_U}."


def _extract_location(text: str) -> str | None:
    import re
    m = re.search(r'\b(?:in|for|at)\s+([A-Za-z\s]+?)(?:\s*\?|$)', text, re.IGNORECASE)
    if m:
        loc = m.group(1).strip()
        if loc.lower() not in ("today", "tomorrow", "now", "the", "a"):
            return loc
    return None


def _geocode(client: httpx.Client, location: str) -> tuple:
    resp = client.get(_GEO_URL, params={"name": location, "count": 1, "language": "en"})
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        raise ValueError(f"Location '{location}' not found")
    r = results[0]
    return r["latitude"], r["longitude"], r.get("name", location)


def _ip_location(client: httpx.Client) -> tuple:
    resp = client.get(_IP_URL)
    resp.raise_for_status()
    d = resp.json()
    return d["lat"], d["lon"], f"{d['city']}, {d['regionName']}"