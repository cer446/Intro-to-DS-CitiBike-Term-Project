import json
import urllib.request
import sys

save_to = None
if len(sys.argv) > 1:
    save_to = sys.argv[1]

newyork = (40.7142,-74.0064)
units = 'auto'

key = '682ab52358ffb984b491906b00547ecb'

base_url = 'https://api.forecast.io/forecast/' \
    '{}/{},{}?units=auto'.format(key, newyork[0], newyork[1])

url = "%s&exclude=%s" % (base_url, 'minutely,hourly,daily,')
print('URL:', url)
response = urllib.request.urlopen(url)
weather = response.read().decode('utf-8')
wjson = json.loads(weather)
time = wjson['currently']['time']

print('Last updated', time)
if not save_to:
    print(json.dumps(bjson, indent=2))
    sys.exit(0)

if '{}' in save_to:
    save_to = save_to.format(time)
print('Saving to', save_to)
with open(save_to, 'w') as f:
    f.write(json.dumps(wjson, indent=2))
