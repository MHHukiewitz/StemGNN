import random

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt


def get_sign_accuracy(target, predict, symbol):
    if 'returns' in symbol:
        predict_symbol = predict[f'{symbol}'] - predict['returnsUSD']
    else:
        predict_symbol = predict[f'{symbol}']
    predict_symbol.name = symbol
    df = target[[f'{symbol}']].join(predict_symbol, rsuffix='_hat')
    acc_df = df.applymap(np.sign)
    acc_df['hit'] = acc_df.apply(lambda x: x[f'{symbol}'] == x[f'{symbol}_hat'], axis=1)
    accuracy = acc_df['hit'].mean()
    print(f"{symbol}: {accuracy:.3f}")
    return accuracy


def analyze_sign_accuracy(name, epoch=None, test=False):
    if test:
        target = pd.read_csv(f"output/{name}/test/target.csv")
        predict = pd.read_csv(f"output/{name}/test/predict.csv")
        if epoch is not None:
            val_target = pd.read_csv(f"output/{name}/train/{epoch}/target.csv")
            val_predict = pd.read_csv(f"output/{name}/train/{epoch}/predict.csv")
            target = pd.concat([val_target, target], axis=0).reset_index()
            predict = pd.concat([val_predict, predict], axis=0).reset_index()
    else:
        if epoch is None:
            epoch = 1
        target = pd.read_csv(f"output/{name}/train/{epoch}/target.csv")
        predict = pd.read_csv(f"output/{name}/train/{epoch}/predict.csv")
    if 'returnsUSD' not in target.columns:
        target['returnsUSD'] = 0.00002
        predict['returnsUSD'] = 0.00002
    total = target.join(predict, rsuffix='_hat')
    total = total.reindex(sorted(total.columns), axis=1)
    print(f"Sign accuracies:")
    log_ret_columns = []
    accs = []
    for column in target.columns:
        acc = get_sign_accuracy(target, predict, column)
        if 'return' in column:
            log_ret_columns.append(column)
            if column != "returnsUSD":
                accs.append(acc)
    accs = pd.Series(accs)
    total['corr'] = predict[log_ret_columns].corrwith(target[log_ret_columns], method='pearson', axis=1)
    total['mag'] = target[log_ret_columns].mean(axis=1).abs()
    total['mw_corr'] = (total['corr'] * total['mag']) / total['mag'].mean()
    print(f"Mean sign accuracy on returns: {accs.mean() * 100: .2f}%")
    print(f"Mean CORR: {total['corr'].mean()} \nMean magnitude: {total['mag'].mean()}")
    print(f"Magnitude weighted (MW) CORR: {total['mw_corr'].mean()}")
    print(f"CORR Sharpe: {total['corr'].mean()/total['corr'].std()}")
    print(f"MW-CORR Sharpe: {total['mw_corr'].mean()/total['mw_corr'].std()}")
    return target[log_ret_columns], predict[log_ret_columns], total


def sigmoid(x, w=0.00001):
    return x*(w+x**2)**-.5


def pred_to_distribution(df: DataFrame, strategy="long", max_leverage=1):
    if strategy == "long/short":
        # normalize
        norm = df.apply(sigmoid)
        # fill NaNs
        norm = norm.ffill().bfill()
        # cap out at max_leverage
        norm['returnsUSD'] = norm['returnsUSD'].values + (1 - norm.sum(axis=1)).values
        for column in norm.columns:
            norm = norm.where(abs(norm[column]) <= max_leverage,
                              (norm * max_leverage).div(norm[column].apply(np.abs), axis=0))
        norm['returnsUSD'] = norm['returnsUSD'].values + (1 - norm.sum(axis=1)).values
        return norm
    else:
        if strategy == "long":
            df = df.apply(lambda row: row - row.min(), axis=1)
            norm = normalize(df)
            crypto_cols = [c for c in norm.columns if 'USD' not in c]
            norm['returnsUSD'] = norm['returnsUSD'] - (norm[crypto_cols].sum(axis=1) * (max_leverage - 1))
            norm[crypto_cols] = norm[crypto_cols] * max_leverage
            return norm.ffill().bfill()
        elif strategy == "short":
            df = df.apply(lambda row: row - row.max(), axis=1)
            norm = -normalize(df)
            crypto_cols = [c for c in df.columns if 'USD' not in c]
            norm['returnsUSD'] = norm['returnsUSD'] - (norm[crypto_cols].sum(axis=1) * (max_leverage - 1))
            norm['returnsUSD'] = norm['returnsUSD'] + 2
            norm[crypto_cols] = norm[crypto_cols] * max_leverage
            return norm.ffill().bfill()


def normalize(df):
    sum = df.sum(axis=1)
    norm = df.div(sum, axis=0)
    return norm

taker_fees = 0.0005
maker_fees = -0.0004
# name = "crypto_vola_244664"  # mit returnsUSD
# name = "crypto_vola_163386"  # 'BTC', 'ETH', 'BNB', 'MATIC', 'ETC', 'ADA'
#name = "crypto_vola_209712"  # 'BTC, 'ETH'
#name = "crypto_vola_279616"  # 'BTC, 'ETH'
#name = "crypto_vola_228666"  # with eth gas
#name = "crypto_vola_219230"  # with eth gas fee $
name = "crypto_vola_1505742"  # all of dem with full BTC history
#name = "crypto_vola_121900"  # all of dem
#name = "crypto_vola_592674"
#name = "crypto_vola_209265"  # without eth gas
#name = "crypto_vola_163386"
real, predict, total = analyze_sign_accuracy(name, 1, test=True)

print(f"\nPORTFOLIO TEST -----------")
base_pf = DataFrame(data=np.zeros((len(real), len(real.columns))), columns=[s[7:] for s in real.columns])
base_pf.iloc[0] = 1000


# TODO: Create a distributed wealth graph for every strategy across all epochs
# TODO: Progressive correlation plotting
# TODO: Plot minimum traded volume
def simulate_even_pf(pf: DataFrame):
    global real
    if 'USD' in pf.columns:  # stay invested in crypto no matter what
        pf.at[0, pf.columns] = pf.iloc[0] + pf.at[0, 'USD'] / (len(pf.columns) - 1)
        pf.at[0, 'USD'] = 0
    print(f"Even portfolio of {pf.iloc[0].sum()}$")
    log_ret_columns = [f"returns{symbol}" for symbol in pf.columns]
    for i in range(1, len(pf)):
        pf.at[i, pf.columns] = pf.iloc[i - 1][pf.columns].values * (np.exp(real.iloc[i - 1][log_ret_columns].values))
        #pf.at[i, pf.columns] = pf.iloc[i][pf.columns].values * 1.000003  # 0,003% market maker
    result, mean, sharpe = get_result(pf)
    return result, mean, sharpe, pf


def get_result(pf, volumes=None):
    global real
    pf['sum'] = pf.sum(axis=1)
    pf['30d_ret'] = pf['sum'].pct_change(30 * 24)
    result = pf['sum'].tail(1).iloc[0]
    print(f"Resulting portfolio value: {result: .2f}$")
    print(f"Return since inception: {((result / pf['sum'].head(1).iloc[0]) - 1) * 100: .2f}%")
    mean_30d = pf['30d_ret'].mean()
    sharpe = mean_30d / pf['30d_ret'].std()
    print(f"Mean 30 days return: {mean_30d * 100: .2f}%, Sharpe: {sharpe: .3f}")
    if volumes is not None:
        print(f"Total volume: {volumes.sum(): .2f}$")
        print(f"Taker fees ({taker_fees * 100: .2f}%): {volumes.sum() * taker_fees/2: .2f}, Maker fees ({maker_fees * 100: .2f}%): {volumes.sum() * maker_fees/2: .2f}")
        print(f"Avg. volume per hour: {volumes.mean(): .2f}$")
    print("------------------------------------------------------")
    return result, mean_30d, sharpe


_, _, _, even_pf = simulate_even_pf(base_pf.copy())
pf = base_pf.copy()
start_capital = base_pf.iloc[0].sum()


def simulate_managed_pf(pf: DataFrame, strategy="long", max_leverage=1):
    global real, predict
    print(f"Managed {strategy} x{max_leverage} portfolio of {pf.iloc[0].sum()}")
    log_ret_columns = [f"returns{symbol}" for symbol in pf.columns]
    crypto_cols = [c for c in pf.columns if c != 'USD']
    norm_predict = pred_to_distribution(predict[log_ret_columns].copy(), strategy=strategy, max_leverage=max_leverage)
    trades = DataFrame()
    for i in range(1, len(pf)):
        # rebalance according to prediction
        pre_rebalance = pf.loc[i - 1, crypto_cols]
        pf.at[i - 1, pf.columns] = pf.iloc[i - 1][pf.columns].sum() * norm_predict.iloc[i - 1].values
        trades.at[i - 1, crypto_cols] = pre_rebalance - pf.loc[i - 1, crypto_cols]
        # apply returns
        pf.at[i, pf.columns] = pf.iloc[i - 1][pf.columns].values * (np.exp(real.iloc[i - 1][log_ret_columns].values))
        multi = np.random.choice([taker_fees, maker_fees])  # random apply taker fee or maker rebate
        pf.at[i, ['USD']] = pf.iloc[i]['USD'] - (trades.loc[i - 1, crypto_cols].abs().sum() * multi)
        #if np.random.randint(0, 1) == 1:
        #    pf.at[i, ['USD']] = pf.iloc[i]['USD'] + (volume_now * 0.0003)  # 0,03% serum maker
        #else:
        #    pf.at[i, ['USD']] = pf.iloc[i]['USD'] - (volume_now * 0.0011)  # 0,03% serum taker
    trades['volume'] = trades.abs().sum(axis=1)
    result, mean, sharpe = get_result(pf, Series(trades['volume']))
    return result, mean, sharpe, pf


alloc = even_pf.copy()


#for i in range(5):
#    _, _, _, l_pf = simulate_managed_pf(base_pf.copy(), strategy="long", max_leverage=i+1)
#    alloc = alloc.join(l_pf, rsuffix=f'_L{i+1}')

for i in range(0, 5, 2):
    _, _, _, ls_pf = simulate_managed_pf(base_pf.copy(), strategy="long/short", max_leverage=i+1)
    alloc = alloc.join(ls_pf, rsuffix=f'_M{i+1}')

#for i in range(5):
#    _, _, _, ls_pf = simulate_managed_pf(base_pf.copy(), strategy="short", max_leverage=i+1)
#    alloc = alloc.join(ls_pf, rsuffix=f'_S{i+1}')

plots = alloc[reversed([c for c in alloc.columns if 'sum' in c])]
plots = plots.applymap(lambda x: x / start_capital)
#plots[[c for c in plots.columns if 'sum' == c or 'L' in c]].plot.line(
#    title=f'LONG {name}: {len(pf)/24: .1f} days', logy=False, colormap='viridis', legend=None)
#plots[[c for c in plots.columns if 'sum' == c or 'S' in c]].plot.line(
#    title=f'SHORT {name}: {len(pf)/24: .1f} days', logy=False, colormap='inferno', legend=None)
plots[[c for c in plots.columns if 'sum' == c or 'M' in c]].plot.line(
    title=f'MIXED {name}: {len(pf)/24: .1f} days', logy=True, colormap='cividis', legend=None)
plt.show()
print("DONE")
