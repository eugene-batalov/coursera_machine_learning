import pandas as pd

content = [x[3:].strip(' \n') for x in open('wine_attrs.txt')]
content = pd.read_csv('wine.data', names=content)

print(content)