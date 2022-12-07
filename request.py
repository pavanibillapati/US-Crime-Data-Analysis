import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'countyCode':0,'communityCode':0,'age':25, 'PctUnemployed':33})

print(r.json())