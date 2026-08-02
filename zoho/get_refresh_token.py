import requests

CLIENT_ID = "1000.X21HMGB6FR6YI9N8T0QJAC6F90YZSP"
CLIENT_SECRET = "a5ef926cb17f67b2e0a1ece96aca60dcaf3cb450f0"
AUTHORIZATION_CODE = "1000.b6348606c0ec18309cf8feaa84f7a644.0b20cbc6c684c2ad3e04fcdc6a3970ff"
REDIRECT_URI = "http://localhost:8000/callback"

url = "https://accounts.zoho.in/oauth/v2/token"

params = {
    "grant_type": "authorization_code",
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "redirect_uri": REDIRECT_URI,
    "code": AUTHORIZATION_CODE,
}

response = requests.post(url, params=params)

print("Status Code:", response.status_code)
print(response.json())