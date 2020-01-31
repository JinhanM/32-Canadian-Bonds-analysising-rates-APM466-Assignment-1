import scipy.interpolate
import pandas as pd
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from numpy import linalg as LA


# read the data file.
xls = pd.ExcelFile('/Users/jinhanmei/Desktop/data for APM466 AS1.xlsx')
df1 = pd.read_excel(xls, '1.2')
df2 = pd.read_excel(xls, '1.3')
df3 = pd.read_excel(xls, '1.6')
df4 = pd.read_excel(xls, '1.7')
df5 = pd.read_excel(xls, '1.8')
df6 = pd.read_excel(xls, '1.9')
df7 = pd.read_excel(xls, '1.10')
df8 = pd.read_excel(xls, '1.13')
df9 = pd.read_excel(xls, '1.14')
df10 = pd.read_excel(xls, '1.15')
df = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]


def time_to_maturity(bond_info):
    current_date = bond_info.columns.values[0]
    bond_info["time to maturity"] = [(maturity - current_date).days for maturity in bond_info["maturity date"]]

def accrued_interest(bond_info):
    temp = []
    for i, bonds in bond_info.iterrows():
        temp.append((182-bonds["time to maturity"] % 182) * bonds["coupon"] * 100 / 365)
    bond_info["accrued interest"] = temp

def dirty_price(bond_info):
    temp = []
    for i, bonds in bond_info.iterrows():
        temp.append(bonds["close price"] + bonds["accrued interest"])
    bond_info["dirty price"] = temp

def yield_calulator(bond_info):
    yield_lst = []
    time_lst = []
    for i, bonds in bond_info.iterrows():
        total_time = bonds["time to maturity"]
        time_lst.append(total_time / 365)

        coupon_times = int(total_time / 182)
        begin_time = (total_time % 182) / 182
        time = np.asarray([begin_time + n for n in range(0, coupon_times + 1)])

        coupon = bonds["coupon"] * 100 / 2
        payments = np.asarray([coupon] * coupon_times + [coupon + 100])

        ytm_func = lambda y: np.dot(payments, (1 + y / 2) ** (-time)) - bonds["dirty price"]

        ytm = optimize.fsolve(ytm_func, .05)
        yield_lst.append(ytm)
    bond_info["yield"] = yield_lst
    bond_info["plot x"] = time_lst

def generate_data(bond_info):
    time_to_maturity(bond_info)
    accrued_interest(bond_info)
    dirty_price(bond_info)
    yield_calulator(bond_info)

def plot_all(all_info):
    labels = ['Jan 2', 'Jan 3', 'Jan 6', 'Jan 7', 'Jan 8', 'Jan 9','Jan 10','Jan 13','Jan 14','Jan 15']
    plt.xlabel('time to maturity')
    plt.ylabel('yield to maturity')
    plt.title('original 5-year yield curve')
    for i in range(len(df)):
        generate_data(all_info[i])
        plt.plot(all_info[i]["plot x"], all_info[i]["yield"], label = labels[i])
    plt.legend(bbox_to_anchor = (0.8,0.98), loc='upper left', borderaxespad=0.)
    plt.show()

plot_all(df)
