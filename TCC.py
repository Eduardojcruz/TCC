#Importar as bibliotecas
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import datetime
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Função para conectar ao MetaTrader 5
def connect_mt5(account, password, server):
    if not mt5.initialize():
        print("Erro: Não foi possível conectar ao MetaTrader 5.")
        return False
    if not mt5.login(account, password, server):
        print(f"Erro: Não foi possível conectar à conta {account}.")
        mt5.shutdown()
        return False
    return True

# Função para obter dados históricos
def get_data(symbol, timeframe, num_candles):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    if rates is None:
        print(f"Erro: Não foi possível obter dados para {symbol} no timeframe {timeframe}.")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# Função para calcular indicadores
def calculate_indicators(df):
    df['MA8'] = df['close'].rolling(window=8).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['Volume_MA'] = df['real_volume'].rolling(window=20).mean()
    df['Volume_above_avg'] = df['real_volume'] > df['Volume_MA']
    df.dropna(inplace=True)
    return df

# Função para preparar dados para LSTM
def prepare_data_lstm(df, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['close']])
    X, y = [], []
    for i in range(len(scaled_data) - time_step - 1):
        X.append(scaled_data[i:(i + time_step), 0])
        y.append(scaled_data[i + time_step, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, scaler

# Função para construir o modelo LSTM
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Função para sincronizar com o fechamento da vela
def wait_for_next_candle(timeframe_minutes=5):
    now = datetime.datetime.now()
    current_minute = (now.minute // timeframe_minutes) * timeframe_minutes
    next_candle = now.replace(minute=current_minute, second=0, microsecond=0) + datetime.timedelta(minutes=timeframe_minutes)
    wait_time = (next_candle - now).total_seconds()
    print(f"Aguardando {wait_time:.2f} segundos para o próximo fechamento de vela ({next_candle.time()})...")
    time.sleep(wait_time)

# Função para decidir trades com base em previsão e indicadores
def decide_trade(df, predicted_price):
    latest = df.iloc[-1]
    if predicted_price > latest['close'] and latest['Volume_above_avg']:
        return 'buy'
    elif predicted_price < latest['close'] and latest['Volume_above_avg']:
        return 'sell'
    return None

# Função para executar uma trade
def execute_trade(symbol, action, volume):
    price = mt5.symbol_info_tick(symbol).ask if action == 'buy' else mt5.symbol_info_tick(symbol).bid
    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': symbol,
        'volume': volume,
        'type': mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL,
        'price': price,
        'sl': 0,
        'tp': 0,
        'magic': 123456,
        'comment': 'trade_rule_based',
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    print(f"Trade Executado: {result}")


conta = int(input("Insira sua conta: "))
senha = input("Insira sua conta: ")
servidor = input("Insira o servidor da conta: ")
ativo = input("Insira o simbolo: ") 

# Configuração inicial
account = conta 
password = senha 
server = servidor  
symbol = ativo
timeframe = mt5.TIMEFRAME_M5  # Timeframe de 5 minutos
volume = 0.1  # Volume do trade

# Conectar ao MetaTrader 5
if not connect_mt5(account, password, server):
    exit()
print("Conectado ao MetaTrader 5 com sucesso!")

# Treinar a LSTM
print("Treinando a LSTM...")
df = get_data(symbol, timeframe, 2000)
df = calculate_indicators(df)
X, y, scaler = prepare_data_lstm(df)
lstm_model = build_lstm_model((X.shape[1], 1))
lstm_model.fit(X, y, batch_size=64, epochs=20)
print("Treinamento concluído!")

# Loop principal
try:
    while True:
        wait_for_next_candle(timeframe_minutes=5)  # Sincronizar com o fechamento da vela
        print("\n--- Iniciando nova análise ---")

        # Coletar e processar dados
        df = get_data(symbol, timeframe, 200)
        if df is None:
            print("Erro ao coletar dados. Tentando novamente.")
            continue
        df = calculate_indicators(df)

        # Fazer previsão com a LSTM
        X_latest, _, _ = prepare_data_lstm(df)
        predicted_price = lstm_model.predict(X_latest[-1].reshape(1, X_latest.shape[1], 1))
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]

        print(f"Preço previsto para a próxima vela: {predicted_price:.2f}")

        # Decidir e executar trade
        action = decide_trade(df, predicted_price)
        if action:
            print(f"Decisão: {action.upper()}")
            execute_trade(symbol, action, volume)
        else:
            print("Nenhuma ação tomada.")

except KeyboardInterrupt:
    print("\nEncerrando o programa por comando do usuário.")
finally:
    mt5.shutdown()
    print("Conexão com MetaTrader 5 encerrada.")
