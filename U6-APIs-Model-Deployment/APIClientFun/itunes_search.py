import requests  # a lib for making http requests
import json  # a lib for parsing strings and json objects

url = 'https://itunes.apple.com/search?'
# add out query terms
url += "term=thor&media=movie"

# make the get request
response = requests.get(url)
# this is a blocking call, so it might take a while to wait for the response
# first check the status code
print("Status code: ", response.status_code)  # if its correct it will be 200
# 200 means ok, everything else is an error
if response.status_code == 200:
    # OK
    # parse the message body JSON
    json_obj = json.loads(response.text)
    print(type(json_obj))  # its a dict :)
    # we want the results list
    results_list = json_obj['results']
    for result_obj in results_list:
        print(result_obj["trackName"])
        # task: print out the movei duration in hours
        milliseconds = result_obj["trackTimeMillis"]
        hours = milliseconds // 3600000
        minutes = (milliseconds % 3600000) // 60000
        seconds = (milliseconds % 60000) // 1000
        print(hours, minutes, seconds)

# task: create interview_predictor.py
# and parse the prediction from the URL given in 3.b.
# NOTE: done :)
