import json
import urllib.request
import sys
import gzip

save_to = None
if len(sys.argv) > 1:
    save_to = sys.argv[1]

url = 'https://gbfs.citibikenyc.com/gbfs/en/station_status.json'

print('URL:', url)
response = urllib.request.urlopen(url)
bikes = response.read().decode('utf-8')
bjson = json.loads(bikes)

print('Last updated', bjson['last_updated'])
if not save_to:
    print(json.dumps(bjson, indent=2))
    sys.exit(0)

if '{}' in save_to:
    save_to = save_to.format(bjson['last_updated'])
print('Saving to', save_to)
with gzip.open(save_to, 'wb') as f:
    f.write(str.encode(json.dumps(bjson)))
