import pandas as pd

cardata ={'mercedas':[1,5,3,5,3],'bmw':[5,8,2,6,5],'ford':[7,8,9,1,0],'reno':[4,9,1,4,7]}
carcatalog=pd.DataFrame(cardata)
carcatalog.index.rename(name='index',inplace=True)
carcatalog.rename(index={0:'ono'},inplace=True)
volvo=[5,8,2,6,5]
carcatalog['volvo']=volvo
pezhu=[13,43,54,76,76]
carcatalog.insert(loc=0 , column='pezhu',value=pezhu)
carcatalog.to_excel("excelexmple.xlsx") #add the Data frame into the excel file
carcatalog_Csvexcel = pd.read_excel("excelexmple.xlsx") #read from the excel file
carcatalog_Csvexcel1 = pd.read_excel("excelexmple.xlsx", index_col = 0) #read from the excel file from first column

print(carcatalog_Csvexcel,carcatalog_Csvexcel1) 