from zoho.books import ZohoBooksClient

client = ZohoBooksClient.from_env()

token = client.get_access_token()

print(token[:20] + "...")