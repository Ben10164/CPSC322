import requests
import json
url = "http://127.0.0.1:5001/"
url += "predict?level=Junior&lang=Jave&tweets=yes&phd=no"

response = requests.get(url)
print("statu code:", response.status_code)
print("response:", response.text)
