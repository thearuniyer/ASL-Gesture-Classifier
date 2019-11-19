import json

path = 'test.json'

with open(path) as j:
    j_data = json.load(j)

print(j_data)
