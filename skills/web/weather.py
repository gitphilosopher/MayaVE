"""
skills/web/weather.py
Weather intent handler for the Maya.

This module resolves a weather request to a specific location, fetches the current
conditions and the day's temperature range from Open-Meteo, and returns a short,
speech-ready string tagged with avatar expressions for downstream playback. If the
user does not provide a place name, it falls back to IP-based geolocation. The
skill is designed to work with the assistant's executor-based async skill model,
where blocking network I/O is deferred away from the event loop and the response is
returned as a plain string.

The implementation keeps the runtime contract simple: it accepts the standard skill
shape (intent, text), extracts the requested locale when present, and formats the
result with a friendly summary that includes temperature, conditions, wind,
humidity, and today's high/low. Time phrases such as "today" or "tomorrow" are
stripped before geocoding so the assistant can still resolve the relevant city
without treating the temporal word as part of the location name.
"""

import asyncio
import logging
import httpx
import re

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
    56: "light freezing drizzle", 57: "freezing drizzle",
    66: "light freezing rain",   67: "heavy freezing rain",
    77: "snow grains", 85: "light snow showers", 86: "heavy snow showers",
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
    56: "sad", 57: "sad", 66: "sad", 67: "angry"
    , 77: "surprised", 85: "surprised", 86: "angry",
}


async def execute(intent: dict, text: str) -> str:
    """Run the weather lookup on the executor thread and return a speech-ready reply."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _fetch, text)


def _fetch(text: str) -> str:
    """Resolve the location, fetch the current forecast, and build the final response."""
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

_TRAILING_TIME_RE = re.compile(
    r"(?:(?:^|\s+)(?:right\s+now|now|today|tonight|tomorrow|"
    r"this\s+(?:morning|afternoon|evening|weekend|week)|please|thanks|thank\s+you))+\s*$",
    re.IGNORECASE,
)

def _extract_location(text: str) -> str | None:
    """Return the requested city or place name while stripping trailing time references."""
    m = re.search(r'\b(?:in|for|at)\s+([A-Za-z\s]+?)\s*[?.!]*$', text, re.IGNORECASE)
    if not m:
        return None
    loc = _TRAILING_TIME_RE.sub("", m.group(1)).strip()   # "London today" -> "London"
    if not loc or loc.lower() in ("the", "a"):
        return None                                        # "for today" -> auto-locate
    return loc

def _geocode(client: httpx.Client, location: str) -> tuple:
    """Look up a place name with the Open-Meteo geocoding API and return coordinates."""
    resp = client.get(_GEO_URL, params={"name": location, "count": 1, "language": "en"})
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        raise ValueError(f"Location '{location}' not found")
    r = results[0]
    return r["latitude"], r["longitude"], r.get("name", location)


def _ip_location(client: httpx.Client) -> tuple:
    """Use the client IP to infer a nearby city and coordinates when no explicit location is given."""
    resp = client.get(_IP_URL)
    resp.raise_for_status()
    d = resp.json()
    return d["lat"], d["lon"], f"{d['city']}, {d['regionName']}"