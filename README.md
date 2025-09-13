# Financial Time Series Bots  

This repository contains implementations for analyzing and predicting financial time series data, including **Nifty 50 stock market data** and **Forex pairs (e.g., XAUUSD, EURUSD)** using different models such as **Permutation Decision Trees (PDT)**, **LSTM**, and **RNN**.  

---

## Repository Structure  

- `pdt_bot.ipynb` – Run Nifty 50 stock strategy using **Permutation Decision Trees (PDT)**.  
- `lstm_bot.ipynb` – Run Nifty 50 stock strategy using an **LSTM model**.  
- `rnn_bot.ipynb` – Run Nifty 50 stock strategy using a **Recurrent Neural Network (RNN)**.  
- `Forex.ipynb` – Run strategy on Forex data (XAUUSD, EURUSD, etc.).  
- `pdtcode.py` and `pdttree.py` – Supporting Python files for PDT logic.  
- `stooq/5 min/world/currencies/major` – Contains Forex time series datasets.  

---

## How to Run  

### 1. Nifty 50 Stocks  

You can experiment with different models for Nifty 50 stock data:  

- **PDT** → Run `pdt_bot.ipynb`  
- **LSTM** → Run `lstm_bot.ipynb`  
- **RNN** → Run `rnn_bot.ipynb`  

Each notebook will guide you through training and evaluating the respective model.  

---

### 2. Forex Data  

1. Open `Forex.ipynb`.  
2. Update the currency pair in **`forex.py`** to either:  
   - `"XAUUSD"` (Gold vs USD)  
   - `"EURUSD"` (Euro vs USD)  
3. Run the notebook to generate results.  
