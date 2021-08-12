import features_test as ft
import pickle
import numpy as np

filename = r"C:\Users\MV\Desktop\Valko\HighFreqMomentumBacktester\NeuralNetwork\finalized_model_EUR_USD.sav"

FX_dict = ft.main()
usd_eur = FX_dict['EUR/USD'].timestamp_base.astype(np.float32).fillna(method="ffill").fillna(
    method="bfill").reset_index()
col_input = ["mid_price", "B_vol", "S_vol", "mid_MA_10", "sum_volume", "UpperBand", "LowerBand"]
loaded_model = pickle.load(open(filename, 'rb'))

# print(loaded_model.predict(usd_eur[col_input]))
value = np.zeros(len(usd_eur))
value[0] = 100
prediction = np.zeros(len(usd_eur))

for index, row in usd_eur.iterrows():
    prediction[index] = loaded_model.predict(np.transpose(row[col_input].values.reshape(-1, 1)))[0]
    try:
        if prediction[index]:
            value[index + 1] = value[index] * (1 + (usd_eur["mid_price"][index + 1] - row["mid_price"]) / row["mid_price"])
        else:
            value[index + 1] = -value[index] * (1 + (usd_eur["mid_price"][index + 1] - row["mid_price"]) / row["mid_price"])
    except KeyError:
        pass

maximal_draw_down = min(value) - 100
pnl = value[-1] - 100

print("Total {0} positions opened.".format(len(value)))
print("Total profit (loss) in percentage is: {0:2.2f}.".format(pnl))
print("Maximal draw down in percentage is: {0:2.6f}.".format(maximal_draw_down))
print("Calmar ratio: {0:2.6f}.".format(pnl/maximal_draw_down))


price_per_pct = 0.001
# We buy then we sell => thus it's traded amount X 2
transaction_price_per_trade = len(value) * 2.0 * price_per_pct / 1000000.00
print("Total transaction price: {0:2.2f}.".format(transaction_price_per_trade * len(value)))
print("Total profit (loss): {0:2.2f}.".format(pnl))
print("Net profit (loss): {0:2.2f}.".format(pnl - transaction_price_per_trade * len(value)))

