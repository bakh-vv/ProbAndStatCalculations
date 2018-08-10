import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import math

sizes = np.array([1300, 1400, 1600, 1900, 2100, 2300])
cost = np.array([88000, 72000, 94000, 86000, 112000, 98000])

plt.scatter(sizes, cost)
plt.show()

file = 'body_image.xls'
xl = pd.ExcelFile(file)
print(xl.sheet_names)
df1 = xl.parse('Sheet1')


df1['GPA'] = pd.to_numeric(df1['GPA'], errors='coerce')
df1['HS_GPA'] = pd.to_numeric(df1['HS_GPA'], errors='coerce')

df1.plot.scatter('HS_GPA', 'GPA')

correlation = df1['GPA'].corr(df1['HS_GPA'])

#cleaning rows where GPA columns contain NaNs (because linear regression don't take in NaNs)
df_clean = df1[np.isfinite(df1['GPA'])]
df_clean = df_clean[np.isfinite(df_clean['HS_GPA'])]

#linear regression fit
z = np.polyfit(df_clean['HS_GPA'], df_clean['GPA'], 1) #returns 0.61922928 (b), 1.07274462 (a)
intercept = z[1]
slope = z[0]

plt.plot(df_clean['HS_GPA'], df_clean['GPA'], 'o', label='original data')
plt.plot(df_clean['HS_GPA'], intercept + slope*df_clean['HS_GPA'], 'r', label='fitted line')
plt.legend()
plt.show()

#two-way frequency table
pd.crosstab(df1['Gender'], df1['WtFeel'])
#with conditional percentages
pd.crosstab(df1['Gender'], df1['WtFeel']).apply(lambda r: r/r.sum(), axis=1)
#alternative syntax
pd.crosstab(df1['Gender'], df1['WtFeel'], normalize='index')
# with subtotals
pd.crosstab(df1['Gender'], df1['WtFeel'], normalize='index', margins=True)

#side-by-side boxplots
df1.boxplot('GPA', by = 'Seat')
df1.groupby('Seat')['GPA'].describe()


###
file2 = 'depression.xls'
xl = pd.ExcelFile(file2)
print(xl.sheet_names)
df2 = xl.parse('Sheet 1')

# Was the randomization effective in assigning an approximately equal number of patients to each treatment 
# group?
df2.groupby("Treat").count()


# Was the randomization successful in balancing other variables such as AcuteT?
df2.boxplot("AcuteT", by = "Treat")
df2.groupby("Treat")["AcuteT"].describe()

# Q1. Which of the drugs (if either) was more successful in preventing the recurrence of depression 
# relative to the placebo?
pd.crosstab(df2["Treat"], df2['Outcome'])
# conditional percentages
pd.crosstab(df2["Treat"], df2['Outcome'], normalize='index')

# Q2. Which of the drugs (if either) delayed the recurrence of depression longer relative to the placebo?
df2.boxplot("Time", by = "Treat")
df2.groupby("Treat")["Time"].describe()


###
#stats.binom.pmf(k, n, p, loc=0)
stats.binom.cdf(4, 10, 0.2)
stats.binom.cdf(2, 10, 0.2)

####

# calculate probability from a z-score
stats.norm.cdf(1.64)
#cdf(value, loc=0, scale=1) # loc= is mean, scale= standard deviation
stats.norm.cdf(65, loc=69, scale=2.8)
stats.norm.cdf(700, loc=507, scale=111)

# calculate z-score from a probability
stats.norm.ppf(.95)
# ppf(q, loc=0, scale=1)
stats.norm.ppf(.98, loc=507, scale=111)
stats.norm.ppf(0.005, loc=69, scale=2.8)
stats.norm.ppf(0.9975, loc=69, scale=2.8)



####
file = 'birthweight.xls'
xl = pd.ExcelFile(file)
print(xl.sheet_names)
df3 = xl.parse('Sheet1')
weight_values= df3['Birthweight']

# confidence interval with population std known
#stats.norm.interval(0.68, loc=mu, scale=sigma/sqrt(N))
stats.norm.interval(0.99, loc=df3['Birthweight'].mean(), scale=500/math.sqrt(df3['Birthweight'].count()))

# confidence interval with population std unknown
file = 'sleep.xls'
xl = pd.ExcelFile(file)
df4 = xl.parse('Sheet1')
sleep_values= df4['Sleep']
#stats.t.interval(0.95, len(g) -1, loc=np.mean(g.Predicted), scale=st.sem(g.Predicted))
# interval(alpha, df, loc=0, scale=1)
stats.t.interval(0.95, len(sleep_values) -1, loc=np.mean(sleep_values), scale=stats.sem(sleep_values))


# confidence interval of population proportion
file = 'guns.xls'
xl = pd.ExcelFile(file)
df5 = xl.parse('Sheet1')
guns_values= df5[0]
guns_values.groupby(guns_values).count()
p = guns_values.groupby(guns_values).count()[1]/guns_values.count()
stats.norm.interval(0.95, loc=p, scale=math.sqrt(p*(1-p)/guns_values.count()))


# calucalting p-value in a z-test for population proportion
p = 0.12658227848
p0 = 0.035
n = 316
1 - stats.norm.cdf(p, loc=p0, scale=math.sqrt(p0*(1-p0)/n))

# calculate z-score
(p-p0)/math.sqrt(p0*(1-p0)/n)

p = 0.19
p0 = 0.157
n = 400
1 - stats.norm.cdf(p, loc=p0, scale=math.sqrt(p0*(1-p0)/n))
stats.norm.sf(p, loc=p0, scale=math.sqrt(p0*(1-p0)/n))

p = 0.16
p0 = 0.20
n = 400
stats.norm.cdf(p, loc=p0, scale=math.sqrt(p0*(1-p0)/n))

p = 0.675
p0 = 0.64
n = 1000
2 * (1- stats.norm.cdf(p, loc=p0, scale=math.sqrt(p0*(1-p0)/n)))

#alternative
from statsmodels.stats import proportion as prop
# prop.proportions_ztest(16, 400, 0.20, alternative='smaller')




# looking at sample distributions through histograms
file = 'time.xls'
xl = pd.ExcelFile(file)
df6 = xl.parse('Sheet1')
df6.hist(column = 'time1')
df6.hist(column = 'time2')
df6.hist(column = 'time3')
df6.hist(column = 'time4')

#Conducting the z-test for the Population Mean. 
x_bar = 550
mu0 = 500
sigma = 100
n = 4
1 - stats.norm.cdf(x_bar, loc=mu0, scale=sigma/math.sqrt(n))

# comparing effects of different sample sizes
df_z_test = pd.DataFrame(columns=['n','z-score','p-value','significant'])

rows_list = []
for i in range(5, 16):
    n = i
    z_score = (x_bar - mu0)/(sigma/math.sqrt(n))
    p_value= 1 - stats.norm.cdf(x_bar, loc=mu0, scale=sigma/math.sqrt(n))
    rows_list.append({"n":n, "z-score":z_score, "p-value":p_value, "significant":p_value<0.05})

df_z_test = pd.DataFrame(rows_list)
df_z_test = df_z_test[['n', 'z-score', 'p-value', 'significant']] #ordering columns in the dataframe


#
file = 'pregnancy.xls'
xl = pd.ExcelFile(file)
df7 = xl.parse('Sheet1')

df7.hist(column = 'Pregnancy Length (days)')

x_bar = df7['Pregnancy Length (days)'].mean()
mu0 = 266
sigma = 16
n = 25
z_score = (x_bar - mu0)/(sigma/math.sqrt(n))
stats.norm.cdf(x_bar, loc=mu0, scale=sigma/math.sqrt(n))




### calculating t statistic
x_bar = 3.93
mu0 = 4.73
n = 75
s = 3.78
t_score = (x_bar - mu0)/(s/math.sqrt(n))

file = 'drinks.xls'
xl = pd.ExcelFile(file)
df8 = xl.parse('Sheet1')

stats.ttest_1samp(df8['number of drinks per week'], mu0)


# data analysis process
file = 'cell_phones.xls'
xl = pd.ExcelFile(file)
df9 = xl.parse('Sheet1')

df9['Verbal'] = pd.to_numeric(df9['Verbal'], errors='coerce')
df9['Verbal'].describe()
df9['Verbal'].hist()

x_bar = 596.713287
mu0 = 580
sigma = 111
n = 286
p_value = 2 *(1 - stats.norm.cdf(x_bar, loc=mu0, scale=sigma/math.sqrt(n)))

stats.norm.interval(0.95, loc=x_bar, scale=sigma/math.sqrt(n))


# data analysis process
df9.groupby('Cell')['Cell'].count()
df9.groupby('Cell')['Cell'].count().plot(kind='pie', autopct='%1.1f%%')

p = 243/310
p0 = 0.8
n = 310
stats.norm.cdf(p, loc=p0, scale=math.sqrt(p0*(1-p0)/n))


# data analysis process
df9['Sleep'] = pd.to_numeric(df9['Sleep'], errors='coerce')
df9['Sleep'].describe()
df9['Sleep'].hist()



x_bar = 3.93
mu0 = 7
n = 75
s = 3.78
t_score = (x_bar - mu0)/(s/math.sqrt(n))

stats.ttest_1samp(df9['Sleep'], mu0)

stats.t.interval(0.95, len(df9['Sleep']) -1, loc=df9['Sleep'].mean(), scale=stats.sem(df9['Sleep']))



# Checking Conditions for the Two-Sample t-test
file = 'lbd1-2.xls'
xl = pd.ExcelFile(file)
df10 = xl.parse('Sheet1')

df10['Time (women)'] = pd.to_numeric(df10['Time (women)'], errors='coerce')
df10['Time (men)'] = pd.to_numeric(df10['Time (men)'], errors='coerce')

df10['Time (women)'].hist()
df10['Time (men)'].hist()


# Carrying Out the Two-Sample t-test
file = 'sleep2.xls'
xl = pd.ExcelFile(file)
df11 = xl.parse('Sheet1')

scipy.stats.ttest_ind(a, b, axis=0, equal_var=False)
stats.ttest_ind(df11['undergraduate'], df11['graduate'], equal_var=False, nan_policy='omit')


# Carrying Out the Paired t-test
file = 'seed.xls'
xl = pd.ExcelFile(file)
df12 = xl.parse('Sheet1')

stats.ttest_rel(df12['Regular seed'], df12['Kiln-dried seed'])
stats.ttest_rel(df12['Regular seed'], df12['Kiln-dried seed']).pvalue / 2

# confidence interval
stats.t.interval(0.95, len(df12['Regular seed'] - df12['Kiln-dried seed']) -1, loc=(df12['Regular seed'] - df12['Kiln-dried seed']).mean(), scale=stats.sem(df12['Regular seed'] - df12['Kiln-dried seed']))



# Carrying Out the ANOVA F-test
file = 'flicker.xls'
xl = pd.ExcelFile(file)
df13 = xl.parse('Sheet1')

df13['Blue'] = pd.to_numeric(df13['Blue'], errors='ignore')
df13['Brown'] = pd.to_numeric(df13['Brown'], errors='ignore')
df13['Green'] = pd.to_numeric(df13['Green'], errors='ignore')
df13['CFF'] = pd.to_numeric(df13['CFF'], errors='coerce')

blue_values = df13['Blue'].dropna()
brown_values = df13['Brown'].dropna()
green_values = df13['Green'].dropna()


df13.boxplot(column = ['Blue', 'Brown', 'Green'])
df13.groupby('Color')['CFF'].describe()

stats.f_oneway(blue_values, brown_values, green_values)



#data analysis process
file = 'gradebook.xls'
xl = pd.ExcelFile(file)
df14 = xl.parse('Sheet1')

df14.boxplot("Final", by = "Extra_Credit")
df14.groupby('Extra_Credit')['Final'].describe()

t_stat = stats.ttest_ind(df14.query('Extra_Credit==0')['Final'], df14.query('Extra_Credit==1')['Final'], equal_var=False, nan_policy='omit').statistic
p_value = stats.ttest_ind(df14.query('Extra_Credit==0')['Final'], df14.query('Extra_Credit==1')['Final'], equal_var=False, nan_policy='omit').pvalue /2


#data analysis process
df14['Diff.Mid'].hist()
df14['Diff.Mid'].describe()

t_stat = stats.ttest_rel(df14['Midterm1'], df14['Midterm2']).statistic
p_value = stats.ttest_rel(df14['Midterm1'], df14['Midterm2']).pvalue / 2


#data analysis process
df14.boxplot("Final", by = "Class")
df14.groupby('Class')['Final'].describe()
df14.query('Class==2')['Final'].hist()

stats.f_oneway(df14.query('Class==1')['Final'], df14.query('Class==2')['Final'], df14.query('Class==3')['Final'], df14.query('Class==4')['Final'])


# Q -> Q / t-test for the slope
file = 'baby-crying-IQ.xls'
xl = pd.ExcelFile(file)
df15 = xl.parse('Sheet1')

plt.scatter(df15['cry count'], df15['IQ'])

stats.linregress(df15['cry count'], df15['IQ'])

f = lambda x: 90.7549885146542 + 1.5363518113635737*x
x = np.array([min(df15['cry count']),max(df15['cry count'])])
plt.plot(x,f(x), c="orange", label="fit line between min and max")

stats.chisquare(a)


# data analysis process
file = 'low_birth_weight.xls'
xl = pd.ExcelFile(file)
df16 = xl.parse('Sheet 1')

pd.crosstab(df16['SMOKE'], df16['LOW'], margins=True)
pd.crosstab(df16['SMOKE'], df16['LOW'], normalize='index', margins=True)

df16[['SMOKE', 'LOW']]
stats.chisquare(df16[['SMOKE', 'LOW']], axis = 1)

stats.chi2_contingency(pd.crosstab(df16['SMOKE'], df16['LOW']), correction = False)

# data analysis process
pd.crosstab(df16['RACE'], df16['LOW'], margins=True)
pd.crosstab(df16['RACE'], df16['LOW'], normalize='index', margins=True)

stats.chi2_contingency(pd.crosstab(df16['RACE'], df16['LOW']), correction = False)

# data analysis process
df16.boxplot('AGE', by = 'LOW')
df16.groupby('LOW')['AGE'].describe()

stats.ttest_ind(df16.query('LOW==0')['AGE'], df16.query('LOW==1')['AGE'], equal_var=False, nan_policy='omit')


# data analysis process
file = 'auto_premiums.xls'
xl = pd.ExcelFile(file)
df17 = xl.parse('Sheet1')

df17.plot.scatter('Experience', 'Premium')
np.corrcoef(df17['Experience'], df17['Premium'])

df17['Experience'].corr(df17['Premium'])

stats.linregress(df17['Experience'], df17['Premium'])

# data anlsysis process
df17.boxplot('Premium', by = 'Gender')
df17.groupby('Gender')['Premium'].describe()


t_stat = stats.ttest_ind(df17.query('Gender==0')['Premium'], df17.query('Gender==1')['Premium'], equal_var=False, nan_policy='omit').statistic
p_value = stats.ttest_ind(df17.query('Gender==0')['Premium'], df17.query('Gender==1')['Premium'], equal_var=False, nan_policy='omit').pvalue / 2
t_stat, p_value
