import scipy.interpolate
import pandas as pd
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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
labels = ['Jan 2','Jan 3','Jan 6','Jan 7','Jan 8', 'Jan 9','Jan 10','Jan 13','Jan 14','Jan 15']


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


# Calculate the yield
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

def generate_data(all_bonds):
    for bond_info in all_bonds:
        time_to_maturity(bond_info)
        accrued_interest(bond_info)
        dirty_price(bond_info)
        yield_calulator(bond_info)


generate_data(df)


# Plot uninterpolated yield curve
def plot_yield(all_info):
    plt.xlabel('time to maturity')
    plt.ylabel('yield to maturity')
    plt.title('original 5-year yield curve')
    for i in range(len(df)):
        plt.plot(all_info[i]["plot x"], all_info[i]["yield"], label = labels[i])
    plt.legend(bbox_to_anchor = (0.8,0.98), loc='upper left', borderaxespad=0.)
    plt.show()


# Interpolation.
def interpolation(x_info, y_info):
    x = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
    temp = []
    inter = interp1d(x_info, y_info, bounds_error=False)
    for i in x:
        value = float(inter(i))
        temp.append(value)
    return np.asarray(x), np.asarray(temp)


# Plot the interpolated yield curve
def plot_yield_inter(all_info):
    plt.xlabel('time to maturity')
    plt.ylabel('yield to maturity')
    plt.title('interpolated 5-year yield curve')
    for i in range(len(all_info)):
        inter_res = interpolation(all_info[i]["plot x"], all_info[i]["yield"])
        plt.plot(inter_res[0], inter_res[1].squeeze(), label = labels[i])
    plt.legend(loc = 'upper right', prop={"size":8})
    plt.show()


#plot_yield_inter(df)


# Calculate the spot rate
def spot_calculator(bond_info):
    s = np.empty([1,10])
    for i, bonds in bond_info.iterrows():
        total_time = bonds["time to maturity"]
        dirty_price = bonds["dirty price"]
        coupons = bonds["coupon"] * 100
        tr = bonds["plot x"]
        if i == 0:
            s[0, i] = -np.log(dirty_price / (coupons / 2 + 100)) / bonds["plot x"]
        else:
            pmt = np.asarray([coupons / 2] * i + [coupons / 2 + 100])
            # print(type(bonds["plot x"][:i]))
            spot_func = lambda y: np.dot(pmt[:-1],
                        np.exp(-(np.multiply(s[0,:i], bond_info["plot x"][:i])))) + pmt[i] * np.exp(-y * bonds["plot x"]) - dirty_price
            s[0, i] = optimize.fsolve(spot_func, .05)
    s[0, 5] = (s[0, 4] + s[0, 6]) / 2
    s[0, 7] = (s[0, 5] + s[0, 8]) / 2
    return s


# Plot the uninterpolated spot curve
def plot_spot(all_info):
    plt.xlabel('time to maturity')
    plt.ylabel('spot rate')
    plt.title('5-year spot curve')
    for i in range(len(all_info)):
        spot_calculator(all_info[i])
        plt.plot(all_info[i]["plot x"], spot(all_info[i]).squeeze(), label = labels[i])
    plt.legend(bbox_to_anchor = (0.8,0.98), loc='upper left', borderaxespad=0.)
    plt.show()


# plot_spot(df)


# Plot the interpolated spot curve
def plot_spot_inter(all_info):
    plt.xlabel('time to maturity')
    plt.ylabel('spot rate')
    plt.title('5-year interpolated spot curve')
    for i in range(len(all_info)):
        spot = spot_calculator(all_info[i])
        x, y = interpolation(all_info[i]["plot x"], spot.squeeze())
        plt.plot(x, y, label = labels[i])
    plt.legend(bbox_to_anchor = (0.8,0.98), loc='upper left', borderaxespad=0.)
    plt.show()


# plot_spot_inter(df)
