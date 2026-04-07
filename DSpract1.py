!pip install pandas
!pip install numpy
!pip install beautifulsoup4
import pandas as pd  
from bs4 import BeautifulSoup  
from urllib.request import urlopen 
url=https://en.wikipedia.org/wiki/List_of_Asian_countries_by_area 
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36  (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/573.36'} 
import requests response=requests.get(url,headers=headers)  
page=response.text print(page)  
soup=BeautifulSoup(page,"html.parser")  
table=soup.find("table") 
print(table)  
SrNo=[]  
Country=[] Area=[] 
rows=table.find("tbody").find_all("tr") 
for row in rows:  
  cells=row.find_all("td")   
  if cells:  
    SrNo.append(cells[0].get_text().strip("\n"))  
    Country.append(cells[1].get_text().strip("\xa0").strip("\n"))  
    Area.append(cells[3].get_text().strip("\n").replace(",","")) 

df=pd.DataFrame()  
df["Area"]=Area 
df.head(10)  



# Web Scrapping
import json 
url=https://jsonplaceholder.typicode.com/users 
from urllib.request import urlopen  
page=urlopen(url) 
data=json.loads(page.read()) 
Id=[]  
Username=[] 
Email=[] 
for item in data:  
  if "id" in item.keys():  
    Id.append(item['id']) 
  else:  
    Id.append("NA") 
  if "username" in item.keys():  
    Username.append(item['username'])  
  else:  
    Username.append("NA")  
  if "email" in item.keys():  
    Email.append(item['email']) 
  else:  
    Email.append("NA") 
df=pd.DataFrame() 
df["Id"]=Id 
df["Username"]=Username 
df["Email"]=Email 
df.head(10)  
df.info()  
