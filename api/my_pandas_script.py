
# viz.head(9).hist()
# plt.show()

# plt.scatter(df.FUELCONSUMPTION_COMB , df.CO2EMISSIONS ,color='blue')
# plt.xlabel("FUEL CONSUMPTION")
# plt.ylabel("CO2 EMISSIONS")
# plt.show()

# plt.scatter(df.CYLINDERS , df.CO2EMISSIONS , color='green')
# plt.xlabel("cylanders")
# plt.ylabel("emission")
# plt.show()

# msk=np.random.rand(len(df) )< 0.8
# train = df[msk]
# test=df[~msk]
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax = plt.scatter(train.ENGINESIZE ,train.CO2EMISSIONS , color='green')
# ax = plt.scatter(test.ENGINESIZE ,test.CO2EMISSIONS , color='red')
# plt.xlabel("engin size")
# plt.ylabel("emission")
# plt.show()

msk=np.random.rand(len(df) )< 0.8
train = df[msk]
test=df[~msk]
regression = linear_model.LinearRegression()
train_x = np.asanyarray(train[["ENGINESIZE"]]) 
train_y =np.asanyarray(train[["CO2EMISSIONS"]])
regression.fit(train_x,train_y)
print('coefficients : tetta 1' , regression.coef_)
print('intercept : tetta 0' , regression.intercept_)

plt.scatter(train.ENGINESIZE ,train.CO2EMISSIONS , color='green')
plt.scatter(train_x ,regression.coef_[0][0]*train_x + regression.intercept_[0] ,color='red' )

plt.xlabel("engin size")
plt.ylabel("emission")
plt.show()

# test : 

test_x = np.asanyarray(train[["ENGINESIZE"]]) 
test_y =np.asanyarray(train[["CO2EMISSIONS"]])

test_y_ = regression.predict(test_x)
 # فرمول های بدست آوردن خطا
print("mean absolut : ", np.mean(np.absolute(test_y_ - test_y)))
print("mean absolut square : ", np.mean(np.absolute(test_y_ - test_y)** 2))
print("r2 score : %.2f : ", r2_score(test_y , test_y_)** 2) # اگر از 0.75 به بالا باشه یعنی خوبه 

print('r2_score:', r2_score(test_y, test_y_))