# https://interview-flask-app.herokuapp.com/predict?level=Junior&lang=Jave&tweets=yes&phd=no

import requests  # a lib for making http requests
import json  # a lib for parsing strings and json objects

url = 'https://interview-flask-app.herokuapp.com/predict?'
# add out query terms
url += "level=Junior&lang=Jave&tweets=yes&phd=no"

# make the get request
response = requests.get(url)
data = json.loads(response.text)
print(data["prediction"])
