list_test =[]
list_untested=['cow','cat','dog','snake']
counter = -1
loop= 10
while len(list_untested) > 0 :
    print(str(list_untested[0]))
    list_test.append(list_untested.pop(0))
    
print(*list_test)

list_untested=[1,2,3,4,5,6]
list_untested = [item*1000 for item in list_untested]
print(list_untested)

list_untested=['cow','cat','dog','snake']
for name in list_untested :
  print(name)

list_untested={'cow':'cat','dog':'snake'}

for key,name in list_untested.items() :
  print(key,name)

list_untested=['cow','cat','dog','snake']
for number,name in enumerate(list_untested) :
  print(number,name)

list_untested=['cow','cat','dog','snake','cow','cat','dog','snake','cow','cat','dog','snake','cow','cat','dog','snake']

for animal in range(1,len(list_untested),2) : 
  print(list_untested[animal])


num=[10,20,30,40]

num = [item*3000 for item in num]

print(num)

def cal(X):
  sum=0.0
  for item in range(len(X)):
    sum=sum+X[item]
    minimum=sum/len(X)
    return minimum
  
minimum = cal(num)
print(minimum)

class mailcat_object():
  def __init__(self , id_name, mail_address):
    self.id_name = id_name
    self.mail_address = mail_address
---------------------------------------------------------------------------
import pickle

class mailcat_object():
  def __init__(self , id_name, mail_address, list_items=[]):
    self.idnum=int
    self.id_name = id_name
    self.mail_address = mail_address
    self.list_items = list_items
    
    def get_it_name(self):
      return self.id_name
    def get_mail_address(self):
      return self.mail_address  
    def get_list_items(self):
      return self.list_items  
    
class mailcat_collection():

  def __init__(self):
      self.mailcat={}
      self.index.counter = 0

  def add_mailcat(self ,mailcat_object):
    self.index.counter += 1
    mailcat_collection.idnum = self.index.counter
    self.mailcat_update({ mailcat_collection.idnum :mailcat_object })

class create_catalog():
  def __init__(self) :
    self.data = mailcat_collection()
    self.data.add_mailcat(mailcat_object("joe", "joe@gmail.com"), [1200, 99, 9, 87, 10, "value", "keys"])
    self.data.add_mailcat(mailcat_object("nd", "sg@gmail.com"), [35, 9864399, 94, 845457, 1110, "valasfue", "keys"])
    self.data.add_mailcat(mailcat_object("joe", "zbxz@gmail.com"), [12, 5435, 9, 8227, 510, "aa", "add"])

    def list_maicat_objcts(self):
      dict = self.data.mailcat
      for key in dict :
        mail=dict[key]
        print(mail.id_name , mail.mai_adress,mail.list_items)
        
    def serialize_mailcat_objects(self):
       dict=self.data.mailcat
       ret_list=[]
       header=['name','mail_adress','int1','fload1','int2']
       ret_list.append(header)
       for key in dict:
         mail=dict[key]
         list_string=''.join([str((item)for item in mail.list_items)])
         mail_string= str(mail.id_name)+""+str(mail.mail_adress)+''+list_string
         ret_list.append(mail_string.split(''))
         
         return ret_list
         
mail_c = create_catalog()
mail_c.list_mailcat_objects()
       
# ---------------------------------------------------------------------------
import csv
import datetime as dt
data = [
    ['sahar', 'sosan'],
    ['sahar', 'sosan'],
    ['sahar', 'sosan']
]

with open("example.csv" , 'w' , newline='') as file :
  writer = csv.writer(file)
  writer.writerows(data)
  
import csv

file_path = r'C:\Users\sahar\OneDrive\Desktop\ML\Text_file.txt'

empty_temp = []
counter = 0 
with open(file_path, 'r', newline='') as file:
    reader_file = csv.reader(file)  
    for row in reader_file:
        print(row)  
        counter +=1
file.close()      
print(counter)


for date in range(len(empty_temp)):
  Birth_date=dt.datetime.strptime(Birth_date , '%Y-%M-%D')
  
  
  print(Birth_date)


Data_X = [0.0, 0.5128205128205128, 1.0256410256410255, 1.5384615384615383, 2.051282051282051, 
     2.564102564102564, 3.0769230769230766, 3.5897435897435894, 4.102564102564102, 
     4.615384615384615, 5.128205128205128, 5.64102564102564, 6.153846153846153, 
     6.666666666666666, 7.179487179487179, 7.692307692307692, 8.205128205128204, 
     8.717948717948717, 9.23076923076923, 9.743589743589743, 10.256410256410255, 
     10.769230769230768, 11.282051282051281, 11.794871794871792, 12.307692307692307, 
     12.82051282051282, 13.333333333333332, 13.846153846153847, 14.35897435897436, 
     14.871794871794872, 15.384615384615383, 15.897435897435898, 16.41025641025641, 
     16.923076923076923, 17.435897435897434, 17.94871794871795, 18.46153846153846, 
     18.97435897435897, 19.487179487179485, 20.0]

Data_Y = [1.0, 0.870130120089978, 0.5200418220162121, 0.05985640109748888, -0.4119558308308628, 
     -0.797320654477022, -0.9923354691509287, -0.9505514906870964, -0.6799782926967007, 
     -0.2393156642875586, 0.2654351720131867, 0.6955820219402375, 0.9510546532543747, 
     0.9463701757149489, 0.6882171490317641, 0.2604056432055987, -0.2532838481985164, 
     -0.6888400300916768, -0.9485613499157301, -0.9519416247557944, -0.6961352386273578, 
     -0.2812700533033356, 0.24103024995736082, 0.6819210208050622, 0.9458251857135236, 
     0.957178261690517, 0.7037282893097331, 0.3018992837195742, -0.22868139875627608, 
     -0.6748283256549617, -0.9428471012265463, -0.9620770981341574, -0.7109929804342464, 
     -0.3222835597381122, 0.21624446030335734, 0.6675653075444877, 0.9396282629785702, 
     0.9666342722815566, 0.7179260409612282, 0.34241300744315094]
Data_X1 = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5]
Data_Y1 = [1.2, 1.7, 2.3, 2.9, 3.5, 4.1, 4.7, 5.3, 5.9, 6.5, 7.1, 7.7, 8.3, 8.9, 9.5, 10.1, 10.7, 11.3, 11.9, 12.5]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(Data_X, Data_Y,color='r')
ax.scatter(Data_X1, Data_Y1, color='b')

min_length = min(len(Data_X), len(Data_X1))

for data in range(min_length):
  X1=(Data_X[data])  
  Y1=(Data_Y[data])
  X2=(Data_X1[data])  
  Y2=(Data_Y1[data])
  ax.plot([X1,X2],[Y1,Y2],color='g')
  
  plt.title("show")
  
plt.show()

import pandas as pd
cardata ={'mercedas':[1,5,3,5,3],'bmw':[5,8,2,6,5],'ford':[7,8,9,1,0],'reno':[4,9,1,4,7]}
carcatalog = pd.DataFrame(cardata)
carcatalog.index.rename(name="index",inplace=True)
carcatalog.rename(index={0:"one"},inplace=True)
newlines=pd.DataFrame({'mercedas':[34,6532]})
carcatalog = pd.concat([carcatalog,newlines])
volvo=[5,8,2,6,5,9,8]
tesla=[8,9,2,3,4,5,2]
pezhu=[13,43,54,76,76,89,9]

carcatalog['volvo']=volvo
carcatalog.insert(loc=5,column="pezhu",value=pezhu)
print(carcatalog)
carcatalog['volvo']= volvo
carcatalog.insert(loc=0, column='tesla',value=tesla)
carspot=[volvo,tesla,pezhu]
carspotcatalog=pd.DataFrame(list(zip(volvo,tesla,pezhu)),columns=["volvo data" , 'tesla data', 'pezhu data'])

carcatalog=pd.DataFrame(cardata)
carcatalog.index.rename(name='index',inplace=True)
carcatalog.rename(index={0:'ono'},inplace=True)
volvo=[5,8,2,6,5]
carcatalog['volvo']=volvo
pezhu=[13,43,54,76,11]
carcatalog.insert(loc=0 , column='pezhu',value=pezhu)
print(carcatalog)



