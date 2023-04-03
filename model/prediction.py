import requests, os, sys
import json
NANONETS_MODEL_ID='8b093def-cfb9-4b0b-931d-5ae417ad8fa5'
NANONETS_API_KEY='f8413371-cfd3-11ed-bc8e-d2e0bcf04a19'
# model_id = os.environ.get('NANONETS_MODEL_ID')
# api_key = os.environ.get('NANONETS_API_KEY')
model_id='8b093def-cfb9-4b0b-931d-5ae417ad8fa5'
api_key='f8413371-cfd3-11ed-bc8e-d2e0bcf04a19'
# image_path = sys.argv[1]
def ocr_text(image_path):

    url = 'https://app.nanonets.com/api/v2/ObjectDetection/Model/' + model_id + '/LabelFile/'

    data = {'file': open(image_path, 'rb'),    'modelId': ('', model_id)}

    response = requests.post(url, auth=requests.auth.HTTPBasicAuth(api_key, ''), files=data)
# print(response.text)
# data = json.loads(response.text)
    response_json = response.json()
# x=data["ocr_text"]
    # print(json.dumps(response_json,))
    json_string = json.dumps(response_json)
    # print(json.dumps(response_json,indent=4))
    data = json.loads(json_string)
    final_number=data['result'][0]['prediction'][0]['ocr_text']
    # print(final_number)
    return final_number
# print(data['result'][0]['prediction'][0]['ocr_text'])

# print()
# print(response.text)