import json
import base64
import requests

# image filepath
IMAGE_FILEPATH = r'C:\Users\vemodalen\Desktop\PRS PROJECT\CK+48\anger\S010_004_00000017.png'

# request
# top_num: result number
PARAMS = {"top_num": 5}

# api from baidu
MODEL_API_URL = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/classification/facial_emotion"

ACCESS_TOKEN = ""
API_KEY = "tTKbtF4QGjRoxhARSqtxABnE"
SECRET_KEY = "OFo6rVZ97HsnZDWdXZztoTYnUGeXBSue"


print("1. read target image '{}'".format(IMAGE_FILEPATH))
with open(IMAGE_FILEPATH, 'rb') as f:
    base64_data = base64.b64encode(f.read())
    base64_str = base64_data.decode('UTF8')
print("fill in 'image' after base63 code")
PARAMS["image"] = base64_str


if not ACCESS_TOKEN:
    print("2. ACCESS_TOKEN is NONE")
    auth_url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials"\
               "&client_id={}&client_secret={}".format(API_KEY, SECRET_KEY)
    auth_resp = requests.get(auth_url)
    auth_resp_json = auth_resp.json()
    ACCESS_TOKEN = auth_resp_json["access_token"]
    print("new ACCESS_TOKEN: {}".format(ACCESS_TOKEN))
else:
    print("2. use existed ACCESS_TOKEN")


print("3. send 'MODEL_API_URL' request")
request_url = "{}?access_token={}".format(MODEL_API_URL, ACCESS_TOKEN)
response = requests.post(url=request_url, json=PARAMS)
response_json = response.json()
response_str = json.dumps(response_json, indent=4, ensure_ascii=False)
print("result:\n{}".format(response_str))