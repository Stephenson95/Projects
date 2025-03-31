import requests
import pandas as pd
import json

base_url = "https://api.domain.com.au/v1/agencies"
url = 'https://api.domain.com.au/sandbox/v1/agencies/22473/listings'

#def get_suburb_info(id):
#    url = f"{base_url}/{id}"

headers = {
    'Accept' : 'application/json',
    'X-API-Key' : 'key_7c9e3d37e20ae193eda56385f5b2312b'
}

params = {
    'listingStatusFilter' : 'live',
    'pageNumber' : 1,
    'pageSize' : 10
}

response = requests.get(url, headers = headers, params = params)
print(response.status_code)
data = json.loads(response.text)

pd.json_normalize(data['data']).head(100)

#auth = HTTPBasicAuth('key_7c9e3d37e20ae193eda56385f5b2312b', '')
#files = {'file': open('filename', 'rb')}

def get_domain_info(url):
    response = requests.get(url, headers = headers)
    if response.status_code == 200:
        data = json.loads(response.text)
        return data
    else:
        return response.status_code



#except (ConnectionError, Timeout, TooManyRedirects) as e:
#    print(e)

#print(data)