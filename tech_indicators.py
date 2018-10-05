import pandas as pd
import numpy as np


bse_data = pd.read_csv('BSESN.csv', header=0, parse_dates=[0])
def rsi(values):
    up = values[values>0].mean()
    down = -1*values[values<0].mean()
    return 100 * up / (up + down)

techindi = bse_data
techindi['Momentum_1D'] = (techindi['Adj Close'] - techindi['Adj Close'].shift(1)).fillna(0)
techindi['RSI_14D'] = techindi['Momentum_1D'].rolling(center=False, window=14).apply(rsi).fillna(0)
techindi['Volume_plain'] = techindi['Volume'].fillna(0)

def bbands(price, length=30, numsd=2):
    """ returns average, upper band, and lower band"""
    #ave = pd.stats.moments.rolling_mean(price,length)
    ave = price.rolling(window = length, center = False).mean()
    #sd = pd.stats.moments.rolling_std(price,length)
    sd = price.rolling(window = length, center = False).std()
    upband = ave + (sd*numsd)
    dnband = ave - (sd*numsd)
    return np.round(ave,3), np.round(upband,3), np.round(dnband,3)

techindi['BB_Middle_Band'], techindi['BB_Upper_Band'], techindi['BB_Lower_Band'] = bbands(techindi['Close'], length=20, numsd=1)
techindi['BB_Middle_Band'] = techindi['BB_Middle_Band'].fillna(0)
techindi['BB_Upper_Band'] = techindi['BB_Upper_Band'].fillna(0)
techindi['BB_Lower_Band'] = techindi['BB_Lower_Band'].fillna(0)

def aroon(df, tf=25):
    aroonup = []
    aroondown = []
    x = tf
    while x< len(df['Date']):
        aroon_up = ((df['High'][x-tf:x].tolist().index(max(df['High'][x-tf:x])))/float(tf))*100
        aroon_down = ((df['Low'][x-tf:x].tolist().index(min(df['Low'][x-tf:x])))/float(tf))*100
        aroonup.append(aroon_up)
        aroondown.append(aroon_down)
        x+=1
    return aroonup, aroondown

listofzeros = [0] * 25
up, down = aroon(techindi)
aroon_list = [x - y for x, y in zip(up,down)]
if len(aroon_list)==0:
    aroon_list = [0] * techindi.shape[0]
    techindi['Aroon_Oscillator'] = aroon_list
else:
    techindi['Aroon_Oscillator'] = listofzeros + aroon_list

techindi["PVT"] = (techindi['Momentum_1D'] / techindi['Close'].shift(1)) * techindi['Volume']
techindi["PVT"] = techindi["PVT"] - techindi["PVT"].shift(1)
techindi["PVT"] = techindi["PVT"].fillna(0)

def STOK(df, n):
    df['STOK'] = ((df['Close'] - df['Low'].rolling(window=n, center=False).mean()) / (df['High'].rolling(window=n, center=False).max() - df['Low'].rolling(window=n, center=False).min())) * 100
    df['STOD'] = df['STOK'].rolling(window = 3, center=False).mean()

columns2Drop = ['Momentum_1D']
techindi = techindi.drop(labels = columns2Drop, axis=1)

STOK(techindi, 4)
techindi = techindi.fillna(0)


def CMFlow(df, tf):
    CHMF = []
    MFMs = []
    MFVs = []
    x = tf

    while x < len(df['Date']):
        PeriodVolume = 0
        volRange = df['Volume'][x - tf:x]
        for eachVol in volRange:
            PeriodVolume += eachVol

        MFM = ((df['Close'][x] - df['Low'][x]) - (df['High'][x] - df['Close'][x])) / (df['High'][x] - df['Low'][x])
        MFV = MFM * PeriodVolume

        MFMs.append(MFM)
        MFVs.append(MFV)
        x += 1

    y = tf
    while y < len(MFVs):
        PeriodVolume = 0
        volRange = df['Volume'][x - tf:x]
        for eachVol in volRange:
            PeriodVolume += eachVol
        consider = MFVs[y - tf:y]
        tfsMFV = 0

        for eachMFV in consider:
            tfsMFV += eachMFV

        tfsCMF = tfsMFV / PeriodVolume
        CHMF.append(tfsCMF)
        y += 1
    return CHMF

listofzeros = [0] * 40
CHMF = CMFlow(techindi, 20)
if len(CHMF)==0:
    CHMF = [0] * techindi.shape[0]
    techindi['Chaikin_MF'] = CHMF
else:
    techindi['Chaikin_MF'] = listofzeros + CHMF

def psar(df, iaf = 0.02, maxaf = 0.2):
    length = len(df)
    dates = (df['Date'])
    high = (df['High'])
    low = (df['Low'])
    close = (df['Close'])
    psar = df['Close'][0:len(df['Close'])]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True
    af = iaf
    ep = df['Low'][0]
    hp = df['High'][0]
    lp = df['Low'][0]
    for i in range(2,length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
        reverse = False
        if bull:
            if df['Low'][i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = df['Low'][i]
                af = iaf
        else:
            if df['High'][i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = df['High'][i]
                af = iaf
        if not reverse:
            if bull:
                if df['High'][i] > hp:
                    hp = df['High'][i]
                    af = min(af + iaf, maxaf)
                if df['Low'][i - 1] < psar[i]:
                    psar[i] = df['Low'][i - 1]
                if df['Low'][i - 2] < psar[i]:
                    psar[i] = df['Low'][i - 2]
            else:
                if df['Low'][i] < lp:
                    lp = df['Low'][i]
                    af = min(af + iaf, maxaf)
                if df['High'][i - 1] > psar[i]:
                    psar[i] = df['High'][i - 1]
                if df['High'][i - 2] > psar[i]:
                    psar[i] = df['High'][i - 2]
        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]
    #return {"dates":dates, "high":high, "low":low, "close":close, "psar":psar, "psarbear":psarbear, "psarbull":psarbull}
    #return psar, psarbear, psarbull
    df['psar'] = psar

psar(techindi)

techindi['ROC'] = ((techindi['Close'] - techindi['Close'].shift(12)) / (techindi['Close'].shift(12))) * 100
techindi = techindi.fillna(0)

techindi['VWAP'] = np.cumsum(techindi['Volume'] * (techindi['High'] + techindi['Low']) / 2) / np.cumsum(
    techindi['Volume'])
techindi = techindi.fillna(0)

techindi['Momentum'] = techindi['Close'] - techindi['Close'].shift(4)
techindi = techindi.fillna(0)

def CCI(df, n, constant):
    TP = (df['High'] + df['Low'] + df['Close']) / 3
    CCI = pd.Series((TP - TP.rolling(window=n, center=False).mean()) / (
    constant * TP.rolling(window=n, center=False).std()))  # , name = 'CCI_' + str(n))
    return CCI

techindi['CCI'] = CCI(techindi, 20, 0.015)
techindi = techindi.fillna(0)

new = (techindi['Volume'] * (~techindi['Close'].diff().le(0) * 2 - 1)).cumsum()
techindi['OBV'] = new

def kelch(df, n):
    KelChM = pd.Series(((df['High'] + df['Low'] + df['Close']) / 3).rolling(window=n, center=False).mean(),
                       name='KelChM_' + str(n))
    KelChU = pd.Series(((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3).rolling(window=n, center=False).mean(),
                       name='KelChU_' + str(n))
    KelChD = pd.Series(((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3).rolling(window=n, center=False).mean(),
                       name='KelChD_' + str(n))
    return KelChM, KelChD, KelChU

KelchM, KelchD, KelchU = kelch(techindi, 14)
techindi['Kelch_Upper'] = KelchU
techindi['Kelch_Middle'] = KelchM
techindi['Kelch_Down'] = KelchD
techindi = techindi.fillna(0)

techindi['HL'] = techindi['High'] - techindi['Low']
techindi['absHC'] = abs(techindi['High'] - techindi['Close'].shift(1))
techindi['absLC'] = abs(techindi['Low'] - techindi['Close'].shift(1))
techindi['TR'] = techindi[['HL', 'absHC', 'absLC']].max(axis=1)
techindi['ATR'] = techindi['TR'].rolling(window=14).mean()
techindi['NATR'] = (techindi['ATR'] / techindi['Close']) * 100
techindi = techindi.fillna(0)

def DMI(df, period):
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['Zero'] = 0

    df['PlusDM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > df['Zero']), df['UpMove'], 0)
    df['MinusDM'] = np.where((df['UpMove'] < df['DownMove']) & (df['DownMove'] > df['Zero']), df['DownMove'], 0)

    df['plusDI'] = 100 * (df['PlusDM']/df['ATR']).ewm(span=period,min_periods=0,adjust=True,ignore_na=False).mean()
    df['minusDI'] = 100 * (df['MinusDM']/df['ATR']).ewm(span=period,min_periods=0,adjust=True,ignore_na=False).mean()

    df['ADX'] = 100 * (abs((df['plusDI'] - df['minusDI'])/(df['plusDI'] + df['minusDI']))).ewm(span=period,min_periods=0,adjust=True,ignore_na=False).mean()

DMI(techindi, 14)
techindi = techindi.fillna(0)

columns2Drop = ['UpMove', 'DownMove', 'ATR', 'PlusDM', 'MinusDM', 'Zero', 'HL', 'absHC', 'absLC', 'TR']
techindi = techindi.drop(labels = columns2Drop, axis=1)

techindi['26_ema'] = techindi['Close'].ewm(span=26, min_periods=0, adjust=True, ignore_na=False).mean()
techindi['12_ema'] = techindi['Close'].ewm(span=12, min_periods=0, adjust=True, ignore_na=False).mean()
techindi['MACD'] = techindi['12_ema'] - techindi['26_ema']
techindi = techindi.fillna(0)


def MFI(df):
    # typical price
    df['tp'] = (df['High'] + df['Low'] + df['Close']) / 3
    # raw money flow
    df['rmf'] = df['tp'] * df['Volume']

    # positive and negative money flow
    df['pmf'] = np.where(df['tp'] > df['tp'].shift(1), df['tp'], 0)
    df['nmf'] = np.where(df['tp'] < df['tp'].shift(1), df['tp'], 0)

    # money flow ratio
    df['mfr'] = df['pmf'].rolling(window=14, center=False).sum() / df['nmf'].rolling(window=14, center=False).sum()
    df['Money_Flow_Index'] = 100 - 100 / (1 + df['mfr'])

MFI(techindi)
techindi = techindi.fillna(0)

def WillR(df):
    highest_high = df['High'].rolling(window=14,center=False).max()
    lowest_low = df['Low'].rolling(window=14,center=False).min()
    df['WillR'] = (-100) * ((highest_high - df['Close']) / (highest_high - lowest_low))

WillR(techindi)
techindi = techindi.fillna(0)

columns2Drop = ['26_ema', '12_ema','tp','rmf','pmf','nmf','mfr']

techindi = techindi.drop(labels = columns2Drop, axis=1)

techindi.to_csv('tech_ind.csv')