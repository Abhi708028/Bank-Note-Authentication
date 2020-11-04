# Advanced-House-Price-Prediction
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization
import matplotlib.pyplot as plt # data visualization
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head() # to check the data
train_df = train.copy()
train.info() # to check what type of data
train.describe()
train.isnull().sum() # to check how many data field is null
# Checking for categorical features

categorical_col = []
for column in train.columns:
    if train[column].dtype == object and len(train[column].unique()) <= 50:
        categorical_col.append(column)
        print(f"{column} : {train[column].unique()}")
        print("====================================")
# Checking for numerical features¶
numerical_col = []
for column in train.columns:
    if train[column].dtype != object and len(train[column].unique()) <= 50:
        numerical_col.append(column)
        print(f"{column} : {train[column].unique()}")
        print("====================================")
# Visulazing the distibution of the data for every feature
train.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20));
# Visualizing the missing values with the help of heatmap.¶
plt.figure(figsize=(20,10))
sns.heatmap(train.isnull(), cmap='viridis')
# Extracting the columns which have missing values from the dataset
missing_values = [feature for feature in train.columns if train[feature].isnull().sum() >1]
    print("The features having the missing values are",missing_values,end='')
for feature in missing_values:
    print(feature, 'has', np.round(train[feature].isnull().mean(),2), '% of missing values')
# Visualizing the Dependent feature
sns.distplot(train['SalePrice'])
train['SalePrice'] = np.log(train['SalePrice'] + 1)
sns.distplot(train['SalePrice'])
# Converting all the categorical columns into numerical¶
print(categorical_col,end='')
for feature in categorical_col:
    temp = train.groupby(feature)['SalePrice'].count()/len(train) #Calculating the percentage
    temp_df = temp[temp>0.01].index
    train[feature] = np.where(train[feature].isin(temp_df), train[feature], 'Rare_var')
train.head()
# Label Encoding the categorical features
# Label encoder basically converts categorical values into numerical values

from sklearn.preprocessing import LabelEncoder

sc=LabelEncoder()

for feature in categorical_col:

    train[feature]=sc.fit_transform(train[feature])
 train.head()
 for feature in missing_values:
    print(feature, 'has', np.round(train[feature].isnull().mean(),2), '% of missing values')
  # Filling the missing values
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())
# Feature Selection
#In our dataset excluding the dependent feature we have 80 indenpendent feature If we consider all the 80 columns as our independent feature our model accuracy will decrease, as the #number of features increases the accuracy decreases this is called as the Curse Of Dimentionality

#In order to solve this problem there are several ways to eliminate this problem like PCA, dropping the useless columns etc.

#But in our case we will use a library under sklearn called as Extra Tree Regressor, what it does is that it returns use only those features which are important for model building, #prediction and the features which helps us it increase the accuracy of the model.

#Feature importance gives you a score for each feature of your data, the higher the score the more important or relevant is the feature towards your output variable

#Feature importance is an in built class that comes with Tree Based Regressor, we will be using Extra Tree Regressor for extracting the top 10 features for the dataset
# Splitting the features into independent and dependent variables

x = train.drop(['SalePrice'], axis = 1)
y = train['SalePrice']
from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()
model.fit(x,y)
print(model.feature_importances_)
# plotting graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()
sns.distplot(train['SalePrice'])
# Building the model
#Spliting data into test and train

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20)
# Applying Linear Regression Algorithm
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train, y_train)

lr_pred = lr.predict(x_test)
r2 = r2_score(y_test,lr_pred)
print('R-Square Score: ',r2*100)
# Calculate the absolute errors
lr_errors = abs(lr_pred - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(lr_pred), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (lr_errors / y_test)
# Calculate and display accuracy
lr_accuracy = 100 - np.mean(mape)
print('Accuracy for Logistic Regression is :', round(lr_accuracy, 2), '%.')
# Accuracy for Logistic Regression is : 99.22 %.
from sklearn.metrics import mean_absolute_error,mean_squared_error

print('mse:',metrics.mean_squared_error(y_test, lr_pred))
print('mae:',metrics.mean_absolute_error(y_test, lr_pred))
# mse: 0.01673922290460145
# mae: 0.09327456017784835
# Applying Decision tree Regressor
from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor(criterion='mse')
dtree.fit(x_train, y_train)
dtree_pred = dtree.predict(x_test)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv
/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt
/kaggle/input/house-prices-advanced-regression-techniques/test.csv
/kaggle/input/house-prices-advanced-regression-techniques/train.csv
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
Data fields
Here's a brief version of what you'll find in the data description file.

SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
MSSubClass: The building class
MSZoning: The general zoning classification
LotFrontage: Linear feet of street connected to property
LotArea: Lot size in square feet
Street: Type of road access
Alley: Type of alley access
LotShape: General shape of property
LandContour: Flatness of the property
Utilities: Type of utilities available
LotConfig: Lot configuration
LandSlope: Slope of property
Neighborhood: Physical locations within Ames city limits
Condition1: Proximity to main road or railroad
Condition2: Proximity to main road or railroad (if a second is present)
BldgType: Type of dwelling
HouseStyle: Style of dwelling
OverallQual: Overall material and finish quality
OverallCond: Overall condition rating
YearBuilt: Original construction date
YearRemodAdd: Remodel date
RoofStyle: Type of roof
RoofMatl: Roof material
Exterior1st: Exterior covering on house
Exterior2nd: Exterior covering on house (if more than one material)
MasVnrType: Masonry veneer type
MasVnrArea: Masonry veneer area in square feet
ExterQual: Exterior material quality
ExterCond: Present condition of the material on the exterior
Foundation: Type of foundation
BsmtQual: Height of the basement
BsmtCond: General condition of the basement
BsmtExposure: Walkout or garden level basement walls
BsmtFinType1: Quality of basement finished area
BsmtFinSF1: Type 1 finished square feet
BsmtFinType2: Quality of second finished area (if present)
BsmtFinSF2: Type 2 finished square feet
BsmtUnfSF: Unfinished square feet of basement area
TotalBsmtSF: Total square feet of basement area
Heating: Type of heating
HeatingQC: Heating quality and condition
CentralAir: Central air conditioning
Electrical: Electrical system
1stFlrSF: First Floor square feet
2ndFlrSF: Second floor square feet
LowQualFinSF: Low quality finished square feet (all floors)
GrLivArea: Above grade (ground) living area square feet
BsmtFullBath: Basement full bathrooms
BsmtHalfBath: Basement half bathrooms
FullBath: Full bathrooms above grade
HalfBath: Half baths above grade
Bedroom: Number of bedrooms above basement level
Kitchen: Number of kitchens
KitchenQual: Kitchen quality
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
Functional: Home functionality rating
Fireplaces: Number of fireplaces
FireplaceQu: Fireplace quality
GarageType: Garage location
GarageYrBlt: Year garage was built
GarageFinish: Interior finish of the garage
GarageCars: Size of garage in car capacity
GarageArea: Size of garage in square feet
GarageQual: Garage quality
GarageCond: Garage condition
PavedDrive: Paved driveway
WoodDeckSF: Wood deck area in square feet
OpenPorchSF: Open porch area in square feet
EnclosedPorch: Enclosed porch area in square feet
3SsnPorch: Three season porch area in square feet
ScreenPorch: Screen porch area in square feet
PoolArea: Pool area in square feet
PoolQC: Pool quality
Fence: Fence quality
MiscFeature: Miscellaneous feature not covered in other categories
MiscVal: $Value of miscellaneous feature
MoSold: Month Sold
YrSold: Year Sold
SaleType: Type of sale
SaleCondition: Condition of sale
train.head()
Id	MSSubClass	MSZoning	LotFrontage	LotArea	Street	Alley	LotShape	LandContour	Utilities	...	PoolArea	PoolQC	Fence	MiscFeature	MiscVal	MoSold	YrSold	SaleType	SaleCondition	SalePrice
0	1	60	RL	65.0	8450	Pave	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	2	2008	WD	Normal	208500
1	2	20	RL	80.0	9600	Pave	NaN	Reg	Lvl	AllPub	...	0	NaN	NaN	NaN	0	5	2007	WD	Normal	181500
2	3	60	RL	68.0	11250	Pave	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	9	2008	WD	Normal	223500
3	4	70	RL	60.0	9550	Pave	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	2	2006	WD	Abnorml	140000
4	5	60	RL	84.0	14260	Pave	NaN	IR1	Lvl	AllPub	...	0	NaN	NaN	NaN	0	12	2008	WD	Normal	250000
5 rows × 81 columns

train_df = train.copy()
train.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 81 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             1460 non-null   int64  
 1   MSSubClass     1460 non-null   int64  
 2   MSZoning       1460 non-null   object 
 3   LotFrontage    1201 non-null   float64
 4   LotArea        1460 non-null   int64  
 5   Street         1460 non-null   object 
 6   Alley          91 non-null     object 
 7   LotShape       1460 non-null   object 
 8   LandContour    1460 non-null   object 
 9   Utilities      1460 non-null   object 
 10  LotConfig      1460 non-null   object 
 11  LandSlope      1460 non-null   object 
 12  Neighborhood   1460 non-null   object 
 13  Condition1     1460 non-null   object 
 14  Condition2     1460 non-null   object 
 15  BldgType       1460 non-null   object 
 16  HouseStyle     1460 non-null   object 
 17  OverallQual    1460 non-null   int64  
 18  OverallCond    1460 non-null   int64  
 19  YearBuilt      1460 non-null   int64  
 20  YearRemodAdd   1460 non-null   int64  
 21  RoofStyle      1460 non-null   object 
 22  RoofMatl       1460 non-null   object 
 23  Exterior1st    1460 non-null   object 
 24  Exterior2nd    1460 non-null   object 
 25  MasVnrType     1452 non-null   object 
 26  MasVnrArea     1452 non-null   float64
 27  ExterQual      1460 non-null   object 
 28  ExterCond      1460 non-null   object 
 29  Foundation     1460 non-null   object 
 30  BsmtQual       1423 non-null   object 
 31  BsmtCond       1423 non-null   object 
 32  BsmtExposure   1422 non-null   object 
 33  BsmtFinType1   1423 non-null   object 
 34  BsmtFinSF1     1460 non-null   int64  
 35  BsmtFinType2   1422 non-null   object 
 36  BsmtFinSF2     1460 non-null   int64  
 37  BsmtUnfSF      1460 non-null   int64  
 38  TotalBsmtSF    1460 non-null   int64  
 39  Heating        1460 non-null   object 
 40  HeatingQC      1460 non-null   object 
 41  CentralAir     1460 non-null   object 
 42  Electrical     1459 non-null   object 
 43  1stFlrSF       1460 non-null   int64  
 44  2ndFlrSF       1460 non-null   int64  
 45  LowQualFinSF   1460 non-null   int64  
 46  GrLivArea      1460 non-null   int64  
 47  BsmtFullBath   1460 non-null   int64  
 48  BsmtHalfBath   1460 non-null   int64  
 49  FullBath       1460 non-null   int64  
 50  HalfBath       1460 non-null   int64  
 51  BedroomAbvGr   1460 non-null   int64  
 52  KitchenAbvGr   1460 non-null   int64  
 53  KitchenQual    1460 non-null   object 
 54  TotRmsAbvGrd   1460 non-null   int64  
 55  Functional     1460 non-null   object 
 56  Fireplaces     1460 non-null   int64  
 57  FireplaceQu    770 non-null    object 
 58  GarageType     1379 non-null   object 
 59  GarageYrBlt    1379 non-null   float64
 60  GarageFinish   1379 non-null   object 
 61  GarageCars     1460 non-null   int64  
 62  GarageArea     1460 non-null   int64  
 63  GarageQual     1379 non-null   object 
 64  GarageCond     1379 non-null   object 
 65  PavedDrive     1460 non-null   object 
 66  WoodDeckSF     1460 non-null   int64  
 67  OpenPorchSF    1460 non-null   int64  
 68  EnclosedPorch  1460 non-null   int64  
 69  3SsnPorch      1460 non-null   int64  
 70  ScreenPorch    1460 non-null   int64  
 71  PoolArea       1460 non-null   int64  
 72  PoolQC         7 non-null      object 
 73  Fence          281 non-null    object 
 74  MiscFeature    54 non-null     object 
 75  MiscVal        1460 non-null   int64  
 76  MoSold         1460 non-null   int64  
 77  YrSold         1460 non-null   int64  
 78  SaleType       1460 non-null   object 
 79  SaleCondition  1460 non-null   object 
 80  SalePrice      1460 non-null   int64  
dtypes: float64(3), int64(35), object(43)
memory usage: 924.0+ KB
train.describe()
Id	MSSubClass	LotFrontage	LotArea	OverallQual	OverallCond	YearBuilt	YearRemodAdd	MasVnrArea	BsmtFinSF1	...	WoodDeckSF	OpenPorchSF	EnclosedPorch	3SsnPorch	ScreenPorch	PoolArea	MiscVal	MoSold	YrSold	SalePrice
count	1460.000000	1460.000000	1201.000000	1460.000000	1460.000000	1460.000000	1460.000000	1460.000000	1452.000000	1460.000000	...	1460.000000	1460.000000	1460.000000	1460.000000	1460.000000	1460.000000	1460.000000	1460.000000	1460.000000	1460.000000
mean	730.500000	56.897260	70.049958	10516.828082	6.099315	5.575342	1971.267808	1984.865753	103.685262	443.639726	...	94.244521	46.660274	21.954110	3.409589	15.060959	2.758904	43.489041	6.321918	2007.815753	180921.195890
std	421.610009	42.300571	24.284752	9981.264932	1.382997	1.112799	30.202904	20.645407	181.066207	456.098091	...	125.338794	66.256028	61.119149	29.317331	55.757415	40.177307	496.123024	2.703626	1.328095	79442.502883
min	1.000000	20.000000	21.000000	1300.000000	1.000000	1.000000	1872.000000	1950.000000	0.000000	0.000000	...	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	1.000000	2006.000000	34900.000000
25%	365.750000	20.000000	59.000000	7553.500000	5.000000	5.000000	1954.000000	1967.000000	0.000000	0.000000	...	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	5.000000	2007.000000	129975.000000
50%	730.500000	50.000000	69.000000	9478.500000	6.000000	5.000000	1973.000000	1994.000000	0.000000	383.500000	...	0.000000	25.000000	0.000000	0.000000	0.000000	0.000000	0.000000	6.000000	2008.000000	163000.000000
75%	1095.250000	70.000000	80.000000	11601.500000	7.000000	6.000000	2000.000000	2004.000000	166.000000	712.250000	...	168.000000	68.000000	0.000000	0.000000	0.000000	0.000000	0.000000	8.000000	2009.000000	214000.000000
max	1460.000000	190.000000	313.000000	215245.000000	10.000000	9.000000	2010.000000	2010.000000	1600.000000	5644.000000	...	857.000000	547.000000	552.000000	508.000000	480.000000	738.000000	15500.000000	12.000000	2010.000000	755000.000000
8 rows × 38 columns

train.isnull().sum()
Id                 0
MSSubClass         0
MSZoning           0
LotFrontage      259
LotArea            0
                ... 
MoSold             0
YrSold             0
SaleType           0
SaleCondition      0
SalePrice          0
Length: 81, dtype: int64
Checking for categorical features
# Checking for categorical features

categorical_col = []
for column in train.columns:
    if train[column].dtype == object and len(train[column].unique()) <= 50:
        categorical_col.append(column)
        print(f"{column} : {train[column].unique()}")
        print("====================================")
MSZoning : ['RL' 'RM' 'C (all)' 'FV' 'RH']
====================================
Street : ['Pave' 'Grvl']
====================================
Alley : [nan 'Grvl' 'Pave']
====================================
LotShape : ['Reg' 'IR1' 'IR2' 'IR3']
====================================
LandContour : ['Lvl' 'Bnk' 'Low' 'HLS']
====================================
Utilities : ['AllPub' 'NoSeWa']
====================================
LotConfig : ['Inside' 'FR2' 'Corner' 'CulDSac' 'FR3']
====================================
LandSlope : ['Gtl' 'Mod' 'Sev']
====================================
Neighborhood : ['CollgCr' 'Veenker' 'Crawfor' 'NoRidge' 'Mitchel' 'Somerst' 'NWAmes'
 'OldTown' 'BrkSide' 'Sawyer' 'NridgHt' 'NAmes' 'SawyerW' 'IDOTRR'
 'MeadowV' 'Edwards' 'Timber' 'Gilbert' 'StoneBr' 'ClearCr' 'NPkVill'
 'Blmngtn' 'BrDale' 'SWISU' 'Blueste']
====================================
Condition1 : ['Norm' 'Feedr' 'PosN' 'Artery' 'RRAe' 'RRNn' 'RRAn' 'PosA' 'RRNe']
====================================
Condition2 : ['Norm' 'Artery' 'RRNn' 'Feedr' 'PosN' 'PosA' 'RRAn' 'RRAe']
====================================
BldgType : ['1Fam' '2fmCon' 'Duplex' 'TwnhsE' 'Twnhs']
====================================
HouseStyle : ['2Story' '1Story' '1.5Fin' '1.5Unf' 'SFoyer' 'SLvl' '2.5Unf' '2.5Fin']
====================================
RoofStyle : ['Gable' 'Hip' 'Gambrel' 'Mansard' 'Flat' 'Shed']
====================================
RoofMatl : ['CompShg' 'WdShngl' 'Metal' 'WdShake' 'Membran' 'Tar&Grv' 'Roll'
 'ClyTile']
====================================
Exterior1st : ['VinylSd' 'MetalSd' 'Wd Sdng' 'HdBoard' 'BrkFace' 'WdShing' 'CemntBd'
 'Plywood' 'AsbShng' 'Stucco' 'BrkComm' 'AsphShn' 'Stone' 'ImStucc'
 'CBlock']
====================================
Exterior2nd : ['VinylSd' 'MetalSd' 'Wd Shng' 'HdBoard' 'Plywood' 'Wd Sdng' 'CmentBd'
 'BrkFace' 'Stucco' 'AsbShng' 'Brk Cmn' 'ImStucc' 'AsphShn' 'Stone'
 'Other' 'CBlock']
====================================
MasVnrType : ['BrkFace' 'None' 'Stone' 'BrkCmn' nan]
====================================
ExterQual : ['Gd' 'TA' 'Ex' 'Fa']
====================================
ExterCond : ['TA' 'Gd' 'Fa' 'Po' 'Ex']
====================================
Foundation : ['PConc' 'CBlock' 'BrkTil' 'Wood' 'Slab' 'Stone']
====================================
BsmtQual : ['Gd' 'TA' 'Ex' nan 'Fa']
====================================
BsmtCond : ['TA' 'Gd' nan 'Fa' 'Po']
====================================
BsmtExposure : ['No' 'Gd' 'Mn' 'Av' nan]
====================================
BsmtFinType1 : ['GLQ' 'ALQ' 'Unf' 'Rec' 'BLQ' nan 'LwQ']
====================================
BsmtFinType2 : ['Unf' 'BLQ' nan 'ALQ' 'Rec' 'LwQ' 'GLQ']
====================================
Heating : ['GasA' 'GasW' 'Grav' 'Wall' 'OthW' 'Floor']
====================================
HeatingQC : ['Ex' 'Gd' 'TA' 'Fa' 'Po']
====================================
CentralAir : ['Y' 'N']
====================================
Electrical : ['SBrkr' 'FuseF' 'FuseA' 'FuseP' 'Mix' nan]
====================================
KitchenQual : ['Gd' 'TA' 'Ex' 'Fa']
====================================
Functional : ['Typ' 'Min1' 'Maj1' 'Min2' 'Mod' 'Maj2' 'Sev']
====================================
FireplaceQu : [nan 'TA' 'Gd' 'Fa' 'Ex' 'Po']
====================================
GarageType : ['Attchd' 'Detchd' 'BuiltIn' 'CarPort' nan 'Basment' '2Types']
====================================
GarageFinish : ['RFn' 'Unf' 'Fin' nan]
====================================
GarageQual : ['TA' 'Fa' 'Gd' nan 'Ex' 'Po']
====================================
GarageCond : ['TA' 'Fa' nan 'Gd' 'Po' 'Ex']
====================================
PavedDrive : ['Y' 'N' 'P']
====================================
PoolQC : [nan 'Ex' 'Fa' 'Gd']
====================================
Fence : [nan 'MnPrv' 'GdWo' 'GdPrv' 'MnWw']
====================================
MiscFeature : [nan 'Shed' 'Gar2' 'Othr' 'TenC']
====================================
SaleType : ['WD' 'New' 'COD' 'ConLD' 'ConLI' 'CWD' 'ConLw' 'Con' 'Oth']
====================================
SaleCondition : ['Normal' 'Abnorml' 'Partial' 'AdjLand' 'Alloca' 'Family']
====================================
Checking for numerical features
numerical_col = []
for column in train.columns:
    if train[column].dtype != object and len(train[column].unique()) <= 50:
        numerical_col.append(column)
        print(f"{column} : {train[column].unique()}")
        print("====================================")
MSSubClass : [ 60  20  70  50 190  45  90 120  30  85  80 160  75 180  40]
====================================
OverallQual : [ 7  6  8  5  9  4 10  3  1  2]
====================================
OverallCond : [5 8 6 7 4 2 3 9 1]
====================================
LowQualFinSF : [  0 360 513 234 528 572 144 392 371 390 420 473 156 515  80  53 232 481
 120 514 397 479 205 384]
====================================
BsmtFullBath : [1 0 2 3]
====================================
BsmtHalfBath : [0 1 2]
====================================
FullBath : [2 1 3 0]
====================================
HalfBath : [1 0 2]
====================================
BedroomAbvGr : [3 4 1 2 0 5 6 8]
====================================
KitchenAbvGr : [1 2 3 0]
====================================
TotRmsAbvGrd : [ 8  6  7  9  5 11  4 10 12  3  2 14]
====================================
Fireplaces : [0 1 2 3]
====================================
GarageCars : [2 3 1 0 4]
====================================
3SsnPorch : [  0 320 407 130 180 168 140 508 238 245 196 144 182 162  23 216  96 153
 290 304]
====================================
PoolArea : [  0 512 648 576 555 480 519 738]
====================================
MiscVal : [    0   700   350   500   400   480   450 15500  1200   800  2000   600
  3500  1300    54   620   560  1400  8300  1150  2500]
====================================
MoSold : [ 2  5  9 12 10  8 11  4  1  7  3  6]
====================================
YrSold : [2008 2007 2006 2009 2010]
====================================
# Visulazing the distibution of the data for every feature
train.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20));

Visualizing the missing values with the help of heatmap.
plt.figure(figsize=(20,10))
sns.heatmap(train.isnull(), cmap='viridis')
<matplotlib.axes._subplots.AxesSubplot at 0x7f16791896d0>

# Extracting the columns which have missing values from the dataset
missing_values = [feature for feature in train.columns if train[feature].isnull().sum() >1]
print("The features having the missing values are",missing_values,end='')
The features having the missing values are ['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
for feature in missing_values:
    print(feature, 'has', np.round(train[feature].isnull().mean(),2), '% of missing values')
LotFrontage has 0.18 % of missing values
Alley has 0.94 % of missing values
MasVnrType has 0.01 % of missing values
MasVnrArea has 0.01 % of missing values
BsmtQual has 0.03 % of missing values
BsmtCond has 0.03 % of missing values
BsmtExposure has 0.03 % of missing values
BsmtFinType1 has 0.03 % of missing values
BsmtFinType2 has 0.03 % of missing values
FireplaceQu has 0.47 % of missing values
GarageType has 0.06 % of missing values
GarageYrBlt has 0.06 % of missing values
GarageFinish has 0.06 % of missing values
GarageQual has 0.06 % of missing values
GarageCond has 0.06 % of missing values
PoolQC has 1.0 % of missing values
Fence has 0.81 % of missing values
MiscFeature has 0.96 % of missing values
We can see that our dataset contains lot of missing values so we need to handle them accordingly
Visualizing the Dependent feature
sns.distplot(train['SalePrice'])
<matplotlib.axes._subplots.AxesSubplot at 0x7f167989b0d0>

We can see that our dependent feature is slightly right skewed this can affect the accuracy of our model so we need to convert it to normal distribution.
train['SalePrice'] = np.log(train['SalePrice'] + 1)
sns.distplot(train['SalePrice'])
<matplotlib.axes._subplots.AxesSubplot at 0x7f167987e650>

We have normalized our dependent feature into Gaussian Distribution to fit our model properly
Converting all the categorical columns into numerical
print(categorical_col,end='')
['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
Now if the percentage is greater than 0.015 then only i am going to take the categorical feature otherwise i am going to say it as rare variable ('Rare_var')
for feature in categorical_col:
    temp = train.groupby(feature)['SalePrice'].count()/len(train) #Calculating the percentage
    temp_df = temp[temp>0.01].index
    train[feature] = np.where(train[feature].isin(temp_df), train[feature], 'Rare_var')
train.head()
Id	MSSubClass	MSZoning	LotFrontage	LotArea	Street	Alley	LotShape	LandContour	Utilities	...	PoolArea	PoolQC	Fence	MiscFeature	MiscVal	MoSold	YrSold	SaleType	SaleCondition	SalePrice
0	1	60	RL	65.0	8450	Pave	Rare_var	Reg	Lvl	AllPub	...	0	Rare_var	Rare_var	Rare_var	0	2	2008	WD	Normal	12.247699
1	2	20	RL	80.0	9600	Pave	Rare_var	Reg	Lvl	AllPub	...	0	Rare_var	Rare_var	Rare_var	0	5	2007	WD	Normal	12.109016
2	3	60	RL	68.0	11250	Pave	Rare_var	IR1	Lvl	AllPub	...	0	Rare_var	Rare_var	Rare_var	0	9	2008	WD	Normal	12.317171
3	4	70	RL	60.0	9550	Pave	Rare_var	IR1	Lvl	AllPub	...	0	Rare_var	Rare_var	Rare_var	0	2	2006	WD	Abnorml	11.849405
4	5	60	RL	84.0	14260	Pave	Rare_var	IR1	Lvl	AllPub	...	0	Rare_var	Rare_var	Rare_var	0	12	2008	WD	Normal	12.429220
5 rows × 81 columns

Label Encoding the categorical features
Label Encoding : Label encoder basically converts categorical values into numerical values
# Label encoder basically converts categorical values into numerical values

from sklearn.preprocessing import LabelEncoder

sc=LabelEncoder()

for feature in categorical_col:

    train[feature]=sc.fit_transform(train[feature])
train.head()
Id	MSSubClass	MSZoning	LotFrontage	LotArea	Street	Alley	LotShape	LandContour	Utilities	...	PoolArea	PoolQC	Fence	MiscFeature	MiscVal	MoSold	YrSold	SaleType	SaleCondition	SalePrice
0	1	60	2	65.0	8450	0	2	3	3	0	...	0	0	3	0	0	2	2008	3	2	12.247699
1	2	20	2	80.0	9600	0	2	3	3	0	...	0	0	3	0	0	5	2007	3	2	12.109016
2	3	60	2	68.0	11250	0	2	0	3	0	...	0	0	3	0	0	9	2008	3	2	12.317171
3	4	70	2	60.0	9550	0	2	0	3	0	...	0	0	3	0	0	2	2006	3	0	11.849405
4	5	60	2	84.0	14260	0	2	0	3	0	...	0	0	3	0	0	12	2008	3	2	12.429220
5 rows × 81 columns

for feature in missing_values:
    print(feature, 'has', np.round(train[feature].isnull().mean(),2), '% of missing values')
LotFrontage has 0.18 % of missing values
Alley has 0.0 % of missing values
MasVnrType has 0.0 % of missing values
MasVnrArea has 0.01 % of missing values
BsmtQual has 0.0 % of missing values
BsmtCond has 0.0 % of missing values
BsmtExposure has 0.0 % of missing values
BsmtFinType1 has 0.0 % of missing values
BsmtFinType2 has 0.0 % of missing values
FireplaceQu has 0.0 % of missing values
GarageType has 0.0 % of missing values
GarageYrBlt has 0.06 % of missing values
GarageFinish has 0.0 % of missing values
GarageQual has 0.0 % of missing values
GarageCond has 0.0 % of missing values
PoolQC has 0.0 % of missing values
Fence has 0.0 % of missing values
MiscFeature has 0.0 % of missing values
Filling the missing values
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())
train.head()
Id	MSSubClass	MSZoning	LotFrontage	LotArea	Street	Alley	LotShape	LandContour	Utilities	...	PoolArea	PoolQC	Fence	MiscFeature	MiscVal	MoSold	YrSold	SaleType	SaleCondition	SalePrice
0	1	60	2	65.0	8450	0	2	3	3	0	...	0	0	3	0	0	2	2008	3	2	12.247699
1	2	20	2	80.0	9600	0	2	3	3	0	...	0	0	3	0	0	5	2007	3	2	12.109016
2	3	60	2	68.0	11250	0	2	0	3	0	...	0	0	3	0	0	9	2008	3	2	12.317171
3	4	70	2	60.0	9550	0	2	0	3	0	...	0	0	3	0	0	2	2006	3	0	11.849405
4	5	60	2	84.0	14260	0	2	0	3	0	...	0	0	3	0	0	12	2008	3	2	12.429220
5 rows × 81 columns

Feature Selection
In our dataset excluding the dependent feature we have 80 indenpendent feature If we consider all the 80 columns as our independent feature our model accuracy will decrease, as the number of features increases the accuracy decreases this is called as the Curse Of Dimentionality

In order to solve this problem there are several ways to eliminate this problem like PCA, dropping the useless columns etc.

But in our case we will use a library under sklearn called as Extra Tree Regressor, what it does is that it returns use only those features which are important for model building, prediction and the features which helps us it increase the accuracy of the model.

Feature importance gives you a score for each feature of your data, the higher the score the more important or relevant is the feature towards your output variable

Feature importance is an in built class that comes with Tree Based Regressor, we will be using Extra Tree Regressor for extracting the top 10 features for the dataset

# Splitting the features into independent and dependent variables

x = train.drop(['SalePrice'], axis = 1)
y = train['SalePrice']
from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()
model.fit(x,y)
ExtraTreesRegressor()
print(model.feature_importances_)
[2.07416514e-03 3.78283565e-03 1.34324035e-02 3.61688221e-03
 7.54406549e-03 2.21083633e-04 1.22485242e-03 2.74694404e-03
 2.73745999e-03 2.14661406e-06 1.61006352e-03 1.98356331e-03
 5.09768856e-03 1.74725569e-03 5.36409826e-04 2.94272946e-03
 1.69074769e-03 1.91294961e-01 7.45129973e-03 3.35544999e-02
 5.16568032e-03 1.99052853e-03 3.60636933e-04 2.20241286e-03
 2.02046085e-03 1.68382004e-03 2.19590241e-03 2.30858754e-01
 2.77673378e-03 1.88301101e-03 1.74135227e-02 1.93476613e-03
 3.14722654e-03 4.03674851e-03 1.00462163e-02 1.21316525e-03
 1.15380730e-03 2.53023069e-03 1.54680590e-02 4.24149783e-04
 2.35279452e-03 3.17687039e-02 1.00248470e-03 1.91465226e-02
 8.86887637e-03 4.97085642e-04 7.66513688e-02 5.53236127e-03
 7.99269117e-04 4.69985443e-02 5.04272241e-03 8.73893951e-03
 1.88071005e-03 1.28365702e-02 6.15134204e-03 2.04526905e-03
 2.29610239e-02 1.88371047e-03 2.46150857e-02 2.41170140e-03
 2.07333660e-02 5.43961911e-02 1.74039207e-02 3.56193984e-03
 2.03061178e-03 2.17658813e-03 2.67402034e-03 3.24097710e-03
 2.22969665e-03 3.48551360e-04 9.58796671e-04 1.83033892e-04
 0.00000000e+00 2.49287019e-03 2.35800988e-04 5.55304728e-04
 2.43717167e-03 2.20820387e-03 1.82008025e-03 4.32989863e-03]
#plotting graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

sns.distplot(train['SalePrice'])
<matplotlib.axes._subplots.AxesSubplot at 0x7f166caeae10>

Building the model
#Spliting data into test and train

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20)
Applying Linear Regression Algorithm
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train, y_train)

lr_pred = lr.predict(x_test)
r2 = r2_score(y_test,lr_pred)
print('R-Square Score: ',r2*100)
R-Square Score:  90.0067359272621
# Calculate the absolute errors
lr_errors = abs(lr_pred - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(lr_pred), 2), 'degrees.')
Mean Absolute Error: 12.04 degrees.
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (lr_errors / y_test)
# Calculate and display accuracy
lr_accuracy = 100 - np.mean(mape)
print('Accuracy for Logistic Regression is :', round(lr_accuracy, 2), '%.')
Accuracy for Logistic Regression is : 99.22 %.
from sklearn.metrics import mean_absolute_error,mean_squared_error

print('mse:',metrics.mean_squared_error(y_test, lr_pred))
print('mae:',metrics.mean_absolute_error(y_test, lr_pred))
mse: 0.01673922290460145
mae: 0.09327456017784835
sns.distplot(y_test-lr_pred)
<matplotlib.axes._subplots.AxesSubplot at 0x7f1679e5c650>

# plotting the Linear Regression values predicated Rating

plt.figure(figsize=(12,7))

plt.scatter(y_test,x_test.iloc[:,2],color="blue")
plt.title("True rate vs Predicted rate",size=20,pad=15)
plt.xlabel('Sale Price',size = 15)
plt.scatter(lr_pred,x_test.iloc[:,2],color="yellow")

# Applying Decision tree Regressor
from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor(criterion='mse')
dtree.fit(x_train, y_train)
DecisionTreeRegressor()
dtree_pred = dtree.predict(x_test)
r2 = r2_score(y_test,dtree_pred)
print('R-Square Score: ',r2*100)

# Calculate the absolute errors
dtree_errors = abs(dtree_pred - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(dtree_pred), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (dtree_errors / y_test)
# Calculate and display accuracy
dtree_accuracy = 100 - np.mean(mape)
print('Accuracy for Decision tree regressor is :', round(dtree_accuracy, 2), '%.')
# R-Square Score:  77.74988533788016
# Mean Absolute Error: 12.06 degrees.
# Accuracy for Decision tree regressor is : 98.87 %.


# plotting the Decision Tree values predicated Rating

plt.figure(figsize=(12,7))

plt.scatter(y_test,x_test.iloc[:,2],color="blue")
plt.title("True rate vs Predicted rate",size=20,pad=15)
plt.xlabel('Sale Price',size = 15)
plt.scatter(dtree_pred,x_test.iloc[:,2],color="yellow")
plt.legend()
# Applying Random Forest Regressor Algorithm
from sklearn.ensemble import RandomForestRegressor

random_forest_regressor = RandomForestRegressor()
random_forest_regressor.fit(x_train, y_train)
rf_pred = random_forest_regressor.predict(x_test)
r2 = r2_score(y_test,rf_pred)
print('R-Square Score: ',r2*100)

# Calculate the absolute errors
rf_errors = abs(rf_pred - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(rf_pred), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (rf_errors / y_test)
# Calculate and display accuracy
rf_accuracy = 100 - np.mean(mape)
print('Accuracy for random forest regressor is :', round(rf_accuracy, 2), '%.')
# R-Square Score:  88.30476373801004
# Mean Absolute Error: 12.04 degrees.
# Accuracy for random forest regressor is : 99.22 %.



# plotting the Random forest values predicated Rating

plt.figure(figsize=(12,7))

plt.scatter(y_test,x_test.iloc[:,2],color="blue")
plt.title("True rate vs Predicted rate",size=20,pad=15)
plt.xlabel('Sale Price',size = 15)
plt.scatter(rf_pred,x_test.iloc[:,2],color="yellow")
