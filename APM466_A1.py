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
    plt.savefig('/Users/jinhanmei/Desktop/APM466HW1/original_yield_curve.png')
    plt.show()


plot_yield(df)


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
    plt.savefig('/Users/jinhanmei/Desktop/APM466HW1/interpolated_yield_curve.png')
    plt.show()



plot_yield_inter(df)


# Calculate the spot rate
def spot_calculator(bond_info):
    s = np.empty([1,11])
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
    plt.title('5-year uninterpolated spot curve')
    for i in range(len(all_info)):
        spot_calculator(all_info[i])
        plt.plot(all_info[i]["plot x"], spot_calculator(all_info[i]).squeeze(), label = labels[i])
    plt.legend(bbox_to_anchor = (0.8,0.98), loc='upper left', borderaxespad=0.)
    plt.savefig('/Users/jinhanmei/Desktop/APM466HW1/original_spot_curve.png')
    plt.show()


plot_spot(df)


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
    plt.savefig('/Users/jinhanmei/Desktop/APM466HW1/interpolated_spot_curve.png')
    plt.show()


plot_spot_inter(df)


# Calculate the forward rate
def forward_calculator(bond_info):
    y = spot_calculator(bond_info).squeeze()
    x, y = interpolation(bond_info["plot x"], y)
    f1 = (y[3] * 2 - y[1] * 1)/(2-1)
    f2 = (y[5] * 3 - y[1] * 1)/(3-1)
    f3 = (y[7] * 4 - y[1] * 1)/(4-1)
    f4 = (y[9] * 5 - y[1] * 1)/(5-1)
    f = [f1,f2,f3,f4]
    return f


# Plot the forward curve
def plot_forward(all_info):
    plt.xlabel('time to maturity')
    plt.ylabel('forward rate')
    plt.title('1-year forward rate curve')
    for i in range(len(all_info)):
        plt.plot(['1yr-1yr','1yr-2yr','1yr-3yr','1yr-4yr'], forward_calculator(all_info[i]), label = labels[i])
    plt.legend(loc = 'upper right', prop={"size":8})
    plt.savefig('/Users/jinhanmei/Desktop/APM466HW1/original_forward_curve.png')
    plt.show()


plot_forward(df)


# calculate covariance matrix
def cov_calculator(all_info):
    log = np.empty([5,9])
    yi = np.empty([5,10])
    for i in range(len(all_info)):
        x,y = interpolation(all_info[i]["plot x"], all_info[i]["yield"])
        yi[0,i] = y[1]
        yi[1,i] = y[3]
        yi[2,i] = y[5]
        yi[3,i] = y[7]
        yi[4,i] = y[9]

    for i in range(0, 9):
        log[0, i] = np.log(yi[0,i+1]/yi[0,i])
        log[1, i] = np.log(yi[1,i+1]/yi[1,i])
        log[2, i] = np.log(yi[2,i+1]/yi[2,i])
        log[3, i] = np.log(yi[3,i+1]/yi[3,i])
        log[4, i] = np.log(yi[4,i+1]/yi[4,i])

    return np.cov(log),log


print("The covariance matrix is: ", cov_calculator(df)[0])


def get_f_matrix(all_info):
    f_m = np.empty([4,10])
    for i in range(len(all_info)):
        f_m[:,i] = forward_calculator(all_info[i])
    return f_m


print("The covariance matrix is: ", np.cov(get_f_matrix(df)))


w1, v1 = LA.eig(np.cov(cov_calculator(df)[1]))
print("The eigenvalue of the matrix is :", w1, "\n,and the eigenvector of the matrix is: ", v1)

w2, v2 = LA.eig(np.cov(get_f_matrix(df)))
print("The eigenvalue of the matrix is :", w2, "\n,and the eigenvector of the matrix is: ", v2)
