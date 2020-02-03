#pusedo code for spot curve

def spot_calculator(bond_info):
    #first create a 1x11 vector.
    s = np.empty([1,11])
    #iter through all bonds
    for i, bonds in bond_info.iterrows():
        #prepare for all the data ready
        prepare_data(bond_info)
        #deal with the case of first coupon date, if it is the first coupon date,
        #we can calculate the spot rate directly.
        if i == 0:
            #use the formula to calculate the first spot rate
            s[0, i] = -np.log(dirty_price / (coupons / 2 + 100)) / bonds["plot x"]
        #then deal with the case of the rest coupon date.
        else:
            #first calculate payments of the bond, and put them in a ndarry
            pmt = np.asarray([coupons / 2] * i + [coupons / 2 + 100])
            #second, set an function about y, according to spot rate formula.
            spot_func = lambda y: np.dot(pmt[:-1],
                        np.exp(-(np.multiply(s[0,:i], bond_info["plot x"][:i]))))
                        + pmt[i] * np.exp(-y * bonds["plot x"]) - dirty_price
            #solve the function.
            s[0, i] = optimize.fsolve(spot_func, .05)
    #in the end, calculate the spot rate at years with missing Mar and Sep data
    s[0, 5] = (s[0, 4] + s[0, 6]) / 2
    s[0, 7] = (s[0, 5] + s[0, 8]) / 2
    return s


#Pseudo code for forward curve.
def forward_calculator(bond_info):
    #first calculate the spot rate at each point.
    y = spot_calculator(bond_info).squeeze()
    #second apply interpolation on points.
    x, y = interpolation(bond_info["plot x"], y)
    #In the end, apply the formula to calculate each 1-1, 1-2, 1-3, 1-4 forwad
    #rate.
    f1 = (y[3] * 2 - y[1] * 1)/(2-1)
    f2 = (y[5] * 3 - y[1] * 1)/(3-1)
    f3 = (y[7] * 4 - y[1] * 1)/(4-1)
    f4 = (y[9] * 5 - y[1] * 1)/(5-1)
    f = [f1,f2,f3,f4]
    return f
