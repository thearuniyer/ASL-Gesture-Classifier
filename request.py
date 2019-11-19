import requests
import json

url = 'http://127.0.0.1:5000/'
path = 'test.json'
# j_data = json.loads(open(path, 'r').read())

with open(path) as j:
    j_data = json.load(j)


headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
print(r, r.text)
