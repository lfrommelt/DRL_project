import re
import os

re.compile(r'.*\.jpg')
for file in os.listdir("data/continuous/"):
    if re.search(r'.*\.jpg', file):
        print(file)
