import urllib.request
import sys
import gzip
import time

save_to = None
if len(sys.argv) > 1:
    save_to = sys.argv[1]

url = 'http://207.251.86.229/nyc-links-cams/LinkSpeedQuery.txt'

print('URL:', url)
now = int(time.time())
response = urllib.request.urlopen(url)
traffic = response.read().decode('utf-8') # csv

if not save_to:
    print(traffic)
    sys.exit(0)

if '{}' in save_to:
    save_to = save_to.format(now)
print('Saving to', save_to)
with open(save_to, 'w') as f:
    f.write(traffic)
