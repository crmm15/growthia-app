import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime

st.set_page_config(page_title="Agent GrowthIA M&M", layout="wide")
st.title("🧠 Plataforma Integral para Gestión y Simulación de Inversiones")

# =========================
# 1. MENÚ LATERAL PRINCIPAL
# =========================
seccion = st.sidebar.radio(
    "📂 Elegí una sección",
    [
        "Inicio",
        "Gestor de Portafolio",
        "Simulador de Opciones",
        "Dashboard de Desempeño",
        "Backtesting Darvas"
    ]
)

# ===============================
# 3. SECCIÓN BACKTESTING DARVAS
# ===============================
if seccion == "Backtesting Darvas":
    st.header("📦 Backtesting Estrategia Darvas Box")

    # Parámetros fijos de los indicadores
    SENSITIVITY = 150
    FAST_EMA = 20
    SLOW_EMA = 40
    CHANNEL_LEN = 20
    BB_MULT = 2.0
    DARVAS_WINDOW = 20  # igual que en la config de TradingView

    # ==============================
    # Funciones auxiliares

    def calc_mavilimw(df, fmal=3, smal=5):
        """Implementa la lógica anidada de medias de MavilimW."""
        M1 = df['Close'].rolling(window=fmal, min_periods=fmal).mean()
        M2 = M1.rolling(window=smal, min_periods=smal).mean()
        M3 = M2.rolling(window=fmal+smal, min_periods=fmal+smal).mean()
        M4 = M3.rolling(window=fmal+2*smal, min_periods=fmal+2*smal).mean()
        M5 = M4.rolling(window=2*fmal+2*smal, min_periods=2*fmal+2*smal).mean()
        return M5  # Última capa, igual al "MAWW" del Pine Script

    def calc_wae(df, sensitivity=150, fastLength=20, slowLength=40, channelLength=20, mult=2.0):
        # MACD de hoy y ayer
        fastMA = df['Close'].ewm(span=fastLength, adjust=False).mean()
        slowMA = df['Close'].ewm(span=slowLength, adjust=False).mean()
        macd = fastMA - slowMA
        macd_shift = macd.shift(1)
        t1 = (macd - macd_shift) * sensitivity

        # Bollinger Bands
        basis = df['Close'].rolling(window=channelLength).mean()
        dev = df['Close'].rolling(window=channelLength).std(ddof=0) * mult
        bb_upper = basis + dev
        bb_lower = basis - dev
        e1 = bb_upper - bb_lower

        # Dead Zone igual que Pine Script: promedio móvil del true range (TR)
        true_range = np.maximum(df['High'] - df['Low'], np.maximum(
            np.abs(df['High'] - df['Close'].shift(1)),
            np.abs(df['Low'] - df['Close'].shift(1))))
        deadzone = pd.Series(true_range).rolling(window=100).mean().fillna(0) * 3.7

        trendUp = np.where(t1 >= 0, t1, 0)
        # trendDown = np.where(t1 < 0, -t1, 0)  # Si alguna vez quieres graficar el histograma bajista

        df['wae_trendUp'] = trendUp
        df['wae_e1'] = e1
        df['wae_deadzone'] = deadzone
        return df

    # Para evitar perder señales iniciales, completamos el filtro de tendencia:
    def robust_trend_filter(df):
        trend = pd.Series(False, index=df.index)
        # Cuando hay valor en mavilimw, tendencia alcista si close > mavilimw (igual que antes)
        trend[df['mavilimw'].notna()] = df.loc[df['mavilimw'].notna(), 'Close'] > df.loc[df['mavilimw'].notna(), 'mavilimw']
        # Para primeras señales: si las últimas 3 velas (incluida la actual) están por arriba de la primera mavilimw válida
        first_valid = df['mavilimw'].first_valid_index()
        if first_valid is not None and first_valid >= 2:
            for i in range(first_valid-1, first_valid+1):
                if i >= 0 and all(df.loc[j, 'Close'] > df.loc[first_valid, 'mavilimw'] for j in range(i, first_valid+1)):
                    trend.iloc[i] = True

        return trend

    # ==============================
    # UI selección
    activos_predef = {
        "BTC/USD": "BTC-USD",
        "ETH/USD": "ETH-USD",
        "Apple (AAPL)": "AAPL",
        "Tesla (TSLA)": "TSLA",
        "Amazon (AMZN)": "AMZN",
        "S&P500 ETF (SPY)": "SPY"
    }
    activo_nombre = st.selectbox("Elige activo para backtesting", list(activos_predef.keys()))
    activo = activos_predef[activo_nombre]

    timeframes = ["1d", "1h", "15m", "5m"]
    timeframe = st.selectbox("Temporalidad", timeframes)

    fecha_inicio = st.date_input("Desde", value=datetime.date(2023, 1, 1), key="darvas_ini")
    fecha_fin = st.date_input("Hasta", value=datetime.date.today(), key="darvas_fin")

    if st.button("Ejecutar Backtest Darvas", key="ejecutar_backtest_darvas"):
        st.info("Descargando datos históricos...")

        df = yf.download(
            activo,
            start=fecha_inicio,
            end=fecha_fin + datetime.timedelta(days=1),
            interval=timeframe,
            progress=False
        )

        if df.empty:
            st.error("No se encontraron datos para ese activo y timeframe. Prueba otra combinación.")
        else:
            st.success(f"Datos descargados: {len(df)} filas")
            st.dataframe(df)

            # Normaliza columnas
            if isinstance(df.columns[0], tuple):
                df.columns = [col[0].capitalize() for col in df.columns]
            else:
                df.columns = [str(col).capitalize() for col in df.columns]
            required_cols = ["Close", "High", "Low"]
            if not all(col in df.columns for col in required_cols):
                st.error(f"El DataFrame descargado NO tiene todas las columnas requeridas: {required_cols}.")
                st.dataframe(df)
            else:
                df = df.reset_index(drop=False)
                df = df.dropna(subset=required_cols)

                # ==============================
                # Indicador Darvas
                df['darvas_high'] = df['High'].rolling(window=DARVAS_WINDOW, min_periods=DARVAS_WINDOW).max()
                df['darvas_low'] = df['Low'].rolling(window=DARVAS_WINDOW, min_periods=DARVAS_WINDOW).min()
                df['prev_darvas_high'] = df['darvas_high'].shift(1)
                df['prev_close'] = df['Close'].shift(1)
                # Señal: solo la PRIMERA ruptura del techo tras consolidación
                df['buy_signal'] = (
                    (df['Close'] > df['prev_darvas_high']) &
                    (df['prev_close'] <= df['prev_darvas_high'])
                )
                df['sell_signal'] = (
                    (df['Close'] < df['darvas_low'].shift(1)) &
                    (df['prev_close'] >= df['darvas_low'].shift(1))
                )

                # ==============================
                # Indicador MavilimW (tendencia)
                df['mavilimw'] = calc_mavilimw(df)

                # Filtro tendencia robusto para primeras velas
                def robust_trend_filter(df):
                    trend = pd.Series(False, index=df.index)
                    trend[df['mavilimw'].notna()] = df.loc[df['mavilimw'].notna(), 'Close'] > df.loc[df['mavilimw'].notna(), 'mavilimw']
                    first_valid = df['mavilimw'].first_valid_index()
                    if first_valid is not None and first_valid >= 1:
                        for i in range(first_valid-1, first_valid+1):
                            if i >= 0 and all(df.loc[j, 'Close'] > df.loc[first_valid, 'mavilimw'] for j in range(i, first_valid+1)):
                                trend.iloc[i] = True
                    return trend

                df['trend_filter'] = robust_trend_filter(df)


                # ==============================
                # Indicador WAE (fuerza/momentum)
                df = calc_wae(
                    df,
                    sensitivity=SENSITIVITY,
                    fastLength=FAST_EMA,
                    slowLength=SLOW_EMA,
                    channelLength=CHANNEL_LEN,
                    mult=BB_MULT
                )
                # Filtro fuerza: solo si el histograma (trendUp) está sobre ExplosionLine y DeadZone
                df['wae_filter'] = (df['wae_trendUp'] > df['wae_e1']) & (df['wae_trendUp'] > df['wae_deadzone'])

                # ==============================
                # Señal final: SOLO cuando las tres condiciones se cumplen
                df['buy_final'] = df['buy_signal'] & df['trend_filter'] & df['wae_filter']

                # ==============================
                # Tabla de señales (solo filas con buy o sell)
                cols_signals = [
                    "Close", "darvas_high", "darvas_low", "mavilimw", "wae_trendUp", "wae_e1", "wae_deadzone",
                    "buy_signal", "trend_filter", "wae_filter", "buy_final", "sell_signal"
                ]
                df_signals = df.loc[df['buy_signal'] | df['sell_signal'], cols_signals].copy()
                num_signals = len(df_signals)
                st.success(f"Número de primeras señales detectadas: {num_signals}")

                st.dataframe(
                    df_signals.head(100),
                    column_config={
                        "Close": st.column_config.NumberColumn("Close", help="Precio de cierre del periodo."),
                        "darvas_high": st.column_config.NumberColumn("darvas_high", help="Máximo de los últimos 20 periodos (techo Darvas)."),
                        "darvas_low": st.column_config.NumberColumn("darvas_low", help="Mínimo de los últimos 20 periodos (base Darvas)."),
                        "mavilimw": st.column_config.NumberColumn("mavilimw", help="Línea MavilimW: tendencia de fondo suavizada (cálculo anidado de medias)."),
                        "wae_trendUp": st.column_config.NumberColumn("wae_trendUp", help="Histograma WAE positivo: fuerza alcista."),
                        "wae_e1": st.column_config.NumberColumn("wae_e1", help="Explosion Line: volatilidad/fuerza según banda de Bollinger."),
                        "wae_deadzone": st.column_config.NumberColumn("wae_deadzone", help="DeadZone: umbral mínimo para considerar fuerza relevante."),
                        "buy_signal": st.column_config.CheckboxColumn("buy_signal", help="True si el cierre rompe el máximo Darvas anterior (solo la primera vez)."),
                        "trend_filter": st.column_config.CheckboxColumn("trend_filter", help="True si la tendencia es alcista (Close > MavilimW)."),
                        "wae_filter": st.column_config.CheckboxColumn("wae_filter", help="True si el histograma supera ambos umbrales de fuerza."),
                        "buy_final": st.column_config.CheckboxColumn("buy_final", help="True si TODAS las condiciones de entrada están OK (ruptura + tendencia + fuerza)."),
                        "sell_signal": st.column_config.CheckboxColumn("sell_signal", help="True si el cierre rompe el mínimo Darvas anterior (solo la primera vez)."),
                    }
                )

                # ==============================
                # Plot gráfico visual
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df.index, df['Close'], label="Precio Close", color="black", zorder=1)
                ax.plot(df.index, df['darvas_high'], label="Darvas High", color="green", linestyle="--", alpha=0.7, zorder=1)
                ax.plot(df.index, df['darvas_low'], label="Darvas Low", color="red", linestyle="--", alpha=0.7, zorder=1)
                ax.plot(df.index, df['mavilimw'], label="MavilimW (Tendencia)", color="white", linewidth=2, zorder=2)
                ax.scatter(df.index[df['buy_final']], df.loc[df['buy_final'], 'Close'], label="Señal Entrada", marker="^", color="blue", s=120, zorder=3)
                ax.scatter(df.index[df['sell_signal']], df.loc[df['sell_signal'], 'Close'], label="Señal Venta", marker="v", color="orange", s=100, zorder=3)
                ax.set_title(f"Darvas Box Backtest - {activo_nombre} [{timeframe}]")
                ax.legend()
                st.pyplot(fig)


# ---- AQUÍ SIGUE TODO EL RESTO DE TU APP ----
# (Gestor de Portafolio, Simulador de Opciones, Dashboard, Inicio, etc)
if seccion == "Inicio":
    st.markdown(open("prompt_inicial.md", "r", encoding="utf-8").read())

def calcular_delta_call_put(S, K, T, r, sigma, tipo="CALL"):
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        if tipo.upper() == "CALL":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    except Exception:
        return None


def generar_y_enviar_resumen_telegram():
    archivo_log = "registro_acciones.csv"
    if not os.path.exists(archivo_log):
        print("⚠ No hay acciones registradas aún.")
        return

    df = pd.read_csv(archivo_log)
    if df.empty:
        print("⚠ El archivo de registro está vacío.")
        return

    resumen = df["Acción Tomada"].value_counts()
    rentabilidad = df.groupby("Acción Tomada")["Rentabilidad %"].mean()

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].pie(resumen, labels=resumen.index, autopct="%1.1f%%", startangle=140)
    axs[0].set_title("Distribución de Decisiones")
    axs[1].bar(rentabilidad.index, rentabilidad.values, color="skyblue")
    axs[1].set_title("Rentabilidad Promedio")
    axs[1].set_ylabel("Rentabilidad %")
    axs[1].tick_params(axis='x', rotation=15)
    plt.tight_layout()
    nombre_archivo = f"resumen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(nombre_archivo)
    plt.close()
    try:
        TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
        TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        with open(nombre_archivo, "rb") as image:
            files = {"photo": image}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": "📊 Resumen de decisiones tomadas"}
            response = requests.post(url, data=data, files=files)
        if response.status_code == 200:
            st.toast("📈 Resumen enviado por Telegram.")
        else:
            st.warning("⚠ No se pudo enviar el gráfico por Telegram.")
    except Exception as e:
        st.warning(f"❌ Error al enviar a Telegram: {e}")
    os.remove(nombre_archivo)

def enviar_grafico_simulacion_telegram(fig, ticker):
    try:
        nombre_archivo = f"simulacion_{ticker}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(nombre_archivo)
        plt.close(fig)
        TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
        TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        with open(nombre_archivo, "rb") as image:
            files = {"photo": image}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": f"📈 Simulación de opción para {ticker}"}
            response = requests.post(url, data=data, files=files)
        if response.status_code == 200:
            st.toast("📤 Simulación enviada por Telegram.")
        else:
            st.warning("⚠ No se pudo enviar el gráfico de simulación por Telegram.")
        os.remove(nombre_archivo)
    except Exception as e:
        st.warning(f"❌ Error al enviar la simulación por Telegram: {e}")

def registrar_accion(ticker, accion, rentab):
    nueva_fila = pd.DataFrame([{
        "Fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Ticker": ticker,
        "Acción Tomada": accion,
        "Rentabilidad %": rentab
    }])
    archivo_log = "registro_acciones.csv"
    if os.path.exists(archivo_log):
        historial = pd.read_csv(archivo_log)
        historial = pd.concat([historial, nueva_fila], ignore_index=True)
    else:
        historial = nueva_fila
    historial.to_csv(archivo_log, index=False)
    try:
        TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
        TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
        mensaje = f"📢 Acción registrada: *{accion}* para `{ticker}` con rentabilidad *{rentab:.2f}%*"
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        params = {"chat_id": TELEGRAM_CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
        requests.get(url, params=params)
        st.toast("📬 Notificación enviada por Telegram.")
    except Exception as e:
        st.warning("⚠ Error al enviar notificación por Telegram.")

def calcular_payoff_call(S, K, premium):
    return np.maximum(S - K, 0) - premium

def calcular_payoff_put(S, K, premium):
    return np.maximum(K - S, 0) - premium

if seccion == "Inicio":
    st.markdown(open("prompt_inicial.md", "r", encoding="utf-8").read())

archivo = st.sidebar.file_uploader("📁 Subí tu archivo Excel (.xlsx)", type=["xlsx"])

# Solo para las secciones que usan el Excel
if seccion in ["Gestor de Portafolio", "Simulador de Opciones", "Dashboard de Desempeño"]:
    if archivo is not None:
        # ... aquí todo tu código de carga y procesamiento del Excel ...
        # ...
        # (nada cambia aquí, todo igual que ya tienes)
        pass
    else:
        st.info("Subí el archivo Excel para empezar.")

# El resto de secciones NO necesitan el Excel cargado, así que no hacen nada aquí


if archivo is not None:
    df = pd.read_excel(archivo, sheet_name="Inversiones")
    df.columns = df.columns.str.strip()
    if 'Ticker' in df.columns and 'Cantidad' in df.columns:
        df = df[df['Ticker'].notnull() & df['Cantidad'].notnull()]

        def limpiar_col_numerica(df, col):
            if col in df.columns:
                temp = (
                    df[col]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .str.replace(" ", "", regex=False)
                    .str.strip()
                )
                return pd.to_numeric(temp, errors="coerce")
            else:
                return np.nan

        for col in ['Rentabilidad', 'Precio Actual', 'DCA', 'Costo', 'Market Value', 'Ganancias/perdidas']:
            df[col] = limpiar_col_numerica(df, col)

        if seccion == "Gestor de Portafolio":
            st.subheader("📊 Análisis de Posiciones")
            for _, row in df.iterrows():
                ticker = row["Ticker"]
                rentab = row["Rentabilidad"]
                precio = row["Precio Actual"]
                dca = row["DCA"]

                if pd.notna(rentab):
                    st.markdown(f"### ▶ {ticker}: {rentab*100:.2f}%")
                else:
                    st.markdown(f"### ▶ {ticker}: nan%")

                if pd.isna(rentab):
                    st.write("🔍 Revisión: Datos incompletos o mal formateados.")
                elif rentab >= 0.2:
                    st.write("🔒 Recomendación: Comprar PUT para proteger ganancias.")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"✅ Ejecutar PUT para {ticker}", key=f"put_{ticker}"):
                            registrar_accion(ticker, "Comprar PUT", rentab)
                            st.success(f"✔ Acción registrada para {ticker}")
                    with col2:
                        if st.button(f"❌ Ignorar recomendación para {ticker}", key=f"ignorar_{ticker}"):
                            registrar_accion(ticker, "Ignorado", rentab)
                            st.info(f"🔕 Recomendación ignorada para {ticker}")
                elif rentab > 0.08:
                    st.write("🔄 Recomendación: Mantener posición.")
                    if st.button(f"✅ Confirmar mantener {ticker}", key=f"mantener_{ticker}"):
                        registrar_accion(ticker, "Mantener", rentab)
                        st.success(f"✔ Acción registrada para {ticker}")
                else:
                    st.write("📉 Recomendación: Revisar, baja rentabilidad.")
                    if st.button(f"📋 Revisar manualmente {ticker}", key=f"revisar_{ticker}"):
                        registrar_accion(ticker, "Revisión Manual", rentab)
                        st.info(f"🔍 Acción registrada para {ticker}")
            st.markdown("---")
            if st.button("📤 Enviar resumen visual a Telegram", key="resumen_telegram"):
                generar_y_enviar_resumen_telegram()

        elif seccion == "Simulador de Opciones":
            st.subheader("📈 Simulador de Opciones con Perfil de Riesgo")

            selected_ticker = st.selectbox("Seleccioná un ticker", df["Ticker"].unique())

            nivel_riesgo = st.radio(
                "🎯 Tu perfil de riesgo",
                ["Conservador", "Balanceado", "Agresivo"],
                index=1,
                help="Define cuánto riesgo estás dispuesto a asumir. Conservador prioriza protección, Agresivo busca mayor upside."
            )

            tipo_opcion = st.radio(
                "Tipo de opción",
                ["CALL", "PUT"],
                help="CALL te beneficia si sube el precio. PUT protege si baja el precio."
            )

            rol = st.radio(
                "Rol en la opción",
                ["Comprador", "Vendedor"],
                index=0,
                help="Elegí si querés simular comprar o vender la opción."
            )

            sugerencia = {"Conservador": 5, "Balanceado": 10, "Agresivo": 20}
            delta_strike = st.slider(
                "📉 % sobre el precio actual para el strike",
                -50, 50, sugerencia[nivel_riesgo],
                help="Determina qué tan alejado estará el strike del precio actual. Positivo para CALL, negativo para PUT."
            )

            dias_a_vencimiento = st.slider(
                "📆 Días hasta vencimiento",
                7, 90, 30,
                help="Número estimado de días hasta la fecha de vencimiento de la opción."
            )

            datos = df[df["Ticker"] == selected_ticker].iloc[0]
            precio_actual = datos["Precio Actual"]
            strike_price = round(precio_actual * (1 + delta_strike / 100), 2)

            ticker_yf = yf.Ticker(selected_ticker)
            expiraciones = ticker_yf.options

            if expiraciones:
                fecha_venc = min(
                    expiraciones,
                    key=lambda x: abs((pd.to_datetime(x) - pd.Timestamp.today()).days - dias_a_vencimiento)
                )

                cadena = ticker_yf.option_chain(fecha_venc)
                tabla_opciones = cadena.calls if tipo_opcion == "CALL" else cadena.puts
                tabla_opciones = tabla_opciones.dropna(subset=["bid", "ask"])

                if tabla_opciones.empty:
                    st.warning("⚠ No hay opciones válidas para ese strike.")
                else:
                    fila = tabla_opciones.loc[np.abs(tabla_opciones["strike"] - strike_price).idxmin()]
                    premium = (fila["bid"] + fila["ask"]) / 2

                st.markdown(f"**Precio actual:** ${precio_actual:.2f}")
                st.markdown(f"**Strike simulado:** ${strike_price}")
                st.markdown(f"**Prima estimada:** ${premium:.2f}")
                st.markdown(f"**Vencimiento elegido:** {fecha_venc}")

                try:
                    if "delta" in fila and not pd.isna(fila["delta"]):
                        delta = fila["delta"]
                    else:
                        T = dias_a_vencimiento / 365
                        r = 0.02
                        sigma = fila.get("impliedVolatility", 0.25)
                        delta = calcular_delta_call_put(precio_actual, strike_price, T, r, sigma, tipo_opcion)

                    if delta is not None:
                        prob = abs(delta) * 100
                        st.markdown(f"**Probabilidad estimada de que se ejecute la opción (Delta): ~{prob:.1f}%**")
                    else:
                        st.warning("⚠ No se pudo calcular el delta estimado.")
                except Exception:
                    st.warning("⚠ Error al calcular el delta.")

                S = np.linspace(precio_actual * 0.6, precio_actual * 1.4, 100)
                payoff = calcular_payoff_call(S, strike_price, premium) if tipo_opcion == "CALL" else calcular_payoff_put(S, strike_price, premium)
                if rol == "Vendedor":
                    payoff = -payoff

                max_payoff = np.max(payoff)
                if premium > 0 and rol == "Comprador":
                    rentabilidad_pct = (max_payoff / premium) * 100
                    st.markdown(f"💰 **Rentabilidad máxima estimada sobre la prima invertida: ~{rentabilidad_pct:.1f}%**")

                break_even = strike_price + premium if tipo_opcion == "CALL" else strike_price - premium
                if rol == "Vendedor":
                    break_even = strike_price - premium if tipo_opcion == "CALL" else strike_price + premium

                fig, ax = plt.subplots(figsize=(5, 3))
                ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
                ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
                ax.set_xlabel("Precio del activo al vencimiento (USD)")
                ax.set_ylabel("Resultado neto (USD)")
                ax.plot(S, payoff, label=f"Payoff ({rol})")
                ax.axhline(0, color="gray", linestyle="--")
                ax.axvline(strike_price, color="red", linestyle="--", label="Strike")
                ax.axvline(break_even, color="green", linestyle="--", label="Break-even")
                ax.set_title(f"{tipo_opcion} - {selected_ticker} ({nivel_riesgo})")
                ax.legend()
                st.pyplot(fig)

                with st.expander("ℹ️ Interpretación del gráfico"):
                    if rol == "Comprador" and tipo_opcion == "CALL":
                        st.markdown(f"🎯 Comprás el derecho a comprar la acción a {strike_price:.2f} pagando una prima de {premium:.2f}")
                        st.markdown("📉 Si el precio final está por debajo del strike, no ejercés y pierdes solo la prima")
                        st.markdown(f"📈 Si el precio sube por encima de {break_even:.2f}, tienes ganancias netas")
                        st.markdown("⚖️ El gráfico muestra tu rentabilidad según el precio al vencimiento")

                    elif rol == "Comprador" and tipo_opcion == "PUT":
                        st.markdown(f"🎯 Comprás el derecho a vender la acción a {strike_price:.2f} pagando una prima de {premium:.2f}")
                        st.markdown(f"📈 Ganás si la acción baja por debajo de {break_even:.2f}")
                        st.markdown("📉 Si se mantiene por encima del strike, la pérdida se limita a la prima")
                        st.markdown("⚖️ El gráfico refleja tu cobertura o especulación a la baja.")

                    elif rol == "Vendedor" and tipo_opcion == "CALL":
                        st.markdown(f"💰 Vendés la opción y recibes {premium:.2f} de prima, pero asumes la obligación de vender a {strike_price:.2f}")
                        st.markdown("✅ Si la acción cierra por debajo del strike, ganás toda la prima")
                        st.markdown(f"⚠️ Si sube por encima de {break_even:.2f}, comenzás a perder dinero")
                        st.markdown("📉 Riesgo ilimitado si el precio sube mucho (al menos que tengas las acciones)")

                    elif rol == "Vendedor" and tipo_opcion == "PUT":
                        st.markdown(f"💰 Vendés la opción y te pagan {premium:.2f} por asumir la obligación de comprar a {strike_price:.2f}")
                        st.markdown("✅ Ganás la prima si el precio se mantiene por encima del strike")
                        st.markdown(f"⚠️ Si cae por debajo de {break_even:.2f}, comenzás a perder dinero")
                        st.markdown("📉 Riesgo limitado: como máximo hasta que la acción llegue a $0")

                with st.expander("📘 Perfil del rol seleccionado"):
                    if rol == "Comprador":
                        st.markdown(f"💸 Pagás una prima {premium:.2f} por el derecho a ejercer")
                        st.markdown("📈 Ganancia potencial ilimitada (CALL) o limitada (PUT)")
                        st.markdown("🔻 Pérdida máxima: la prima")
                    else:
                        if tipo_opcion == "CALL":
                            st.markdown(f"💵 Recibes una prima {premium:.2f} por asumir la obligación de vender a {strike_price:.2f}")
                            st.markdown("✅ Ganancia máxima: la prima si la acción no supera el strike")
                            st.markdown(f"⚠️ Si el precio sube por encima de {break_even:.2f}, comenzás a tener pérdidas. Estas son potencialmente ilimitadas")
                            st.markdown("🔒 Estrategia útil para generar ingresos si creés que la acción no superará el strike")
                        else:
                            st.markdown(f"💵 Recibes una prima {premium:.2f} por asumir la obligación de comprar a {strike_price:.2f}")
                            st.markdown("✅ Ganancia máxima: la prima si la acción se mantiene por encima del strike.")
                            st.markdown(f"⚠️ Si la acción cae por debajo de {break_even:.2f}, empiezás a tener pérdidas. El riesgo es alto, pero finito (hasta que la acción llegue a $0)")
                            st.markdown("🛡 Estrategia usada si estás dispuesto a comprar la acción más barata que hoy")

                if st.button("📤 Enviar esta simulación a Telegram", key="Enviar_Simulación"):
                    enviar_grafico_simulacion_telegram(fig, selected_ticker)

            else:
                st.warning("⚠ No se encontró cadena de opciones para este ticker.")

        elif seccion == "Dashboard de Desempeño":
            try:
                historial = pd.read_csv("registro_acciones.csv")
                historial["Fecha"] = pd.to_datetime(historial["Fecha"])
                tickers = historial["Ticker"].unique()
                filtro = st.multiselect("📌 Filtrar Tickers", options=tickers, default=list(tickers))
                df_filtrado = historial[historial["Ticker"].isin(filtro)]

                st.subheader("📈 Indicadores Generales")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total decisiones", len(df_filtrado))
                col2.metric("% PUTs", f"{(df_filtrado['Acción Tomada'] == 'Comprar PUT').mean() * 100:.1f}%")
                col3.metric("% Mantener", f"{(df_filtrado['Acción Tomada'] == 'Mantener').mean() * 100:.1f}%")

                st.bar_chart(df_filtrado.groupby("Acción Tomada")["Rentabilidad %"].mean())
                st.line_chart(df_filtrado.set_index("Fecha")["Rentabilidad %"])
            except FileNotFoundError:
                st.error("No se encontró 'registro_acciones.csv'. Ejecutá primero el gestor.")


# --- Envío automático del resumen diario por Telegram a las 23hs ---
# from datetime import datetime
# ahora = datetime.now()
# if ahora.hour == 23 and ahora.minute < 5:
#     generar_y_enviar_resumen_telegram()
#     st.toast("📤 Resumen diario enviado automáticamente.")





