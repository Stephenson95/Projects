import requests
import pandas as pd
import json

#Api-key
url = 'https://api.domain.com.au/sandbox/v1/agencies/22473/listings'

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
listings = json.loads(response.text)

output = pd.DataFrame()
for index, prop_list in enumerate(listings):
    result = {'objective':prop_list['objective'],
              'propertytype':prop_list['propertyTypes'][0],
              'status':prop_list['status'],
              'channel':prop_list['channel'],
              'state':prop_list['addressParts']['stateAbbreviation'],
              'streetname':prop_list['addressParts']['street'],
              'suburb':prop_list['addressParts']['suburb'],
              'postcode':prop_list['addressParts']['postcode'],
              'bathrooms':prop_list['bathrooms'],
              'bedrooms':prop_list['bedrooms'],
              'carspaces':prop_list['carspaces'],
              'price':prop_list['priceDetails']['displayPrice'],
              'datelisted':prop_list['dateListed'],
              'latitude':prop_list['geoLocation']['latitude'],
              'longitude':prop_list['geoLocation']['longitude'],
              'newdevelopment':prop_list['isNewDevelopment']}
    
    if 'unitNumber' in prop_list.keys():
        result['unitnumber'] = prop_list['addressParts']['unitNumber']
    
    if 'streetNumber' in prop_list.keys():
        result['streetnumber'] = prop_list['addressParts']['streetNumber']
    
    if 'landAreaSqm' in prop_list.keys():
        result['landsize'] = prop_list['landAreaSqm']
        
    if 'buildingAreaSqm' in prop_list.keys():
        result['buildingsize'] = prop_list['buildingAreaSqm']
        
    if 'features' in prop_list.keys():
        result['features'] = prop_list['features']
    
    output = pd.concat([output, pd.DataFrame.from_dict(result, orient = 'index').T])
    


def get_domain_info(url):
    response = requests.get(url, headers = headers)
    if response.status_code == 200:
        data = json.loads(response.text)
        return data
    else:
        return response.status_code



#%%
#OAuth2.0
auth = ('client_9ae3ca1968db2a5ff42df36673301b94', 'secret_1b8014271e61bf92e9d280f1146b5cf0')

#obtain access token
token_url = "https://auth.domain.com.au/v1/connect/token"
response = requests.post(token_url, auth=auth, data = {"grant_type" : "client_credentials",
                                                       "scope":"api_listings_read"})
#print(json.dumps(response.json(),indent=4))
access_token = response.json()["access_token"]
token_type = response.json()["token_type"]

url = 'https://api.domain.com.au/sandbox/v1/agencies/22473/listings'

headers = {
    'Accept' : 'application/json',
    'Authorization' : token_type + " " + access_token
}

params = {
    'listingStatusFilter' : 'live',
    'pageNumber' : 1,
    'pageSize' : 10
}

response = requests.get(url, headers = headers, params = params)
print(response.status_code)
listings = json.loads(response.text)