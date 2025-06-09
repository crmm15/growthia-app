
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import datetime
import os
import requests  # Para enviar mensajes a Telegram

st.set_page_config(page_title="Agent GrowthIA M&M", layout="wide")
st.title("🧠 Plataforma Integral para Gestión y Simulación de Inversiones")

from scipy.stats import norm

from scipy.stats import norm
import math

def calcular_delta_call_put(S, K, T, r, sigma, tipo="CALL"):
    """
    Calcula el delta de una opción usando Black-Scholes.
    S = precio actual del activo
    K = precio de ejercicio (strike)
    T = tiempo hasta vencimiento (en años)
    r = tasa libre de riesgo (anual)
    sigma = volatilidad (anual)
    tipo = "CALL" o "PUT"
    """
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        if tipo.upper() == "CALL":
            return norm.cdf(d1)
        else:  # PUT
            return norm.cdf(d1) - 1
    except Exception as e:
        return None


# Menú principal
seccion = st.sidebar.radio("📂 Elegí una sección", ["Inicio", "Gestor de Portafolio", "Simulador de Opciones", "Dashboard de Desempeño"])

def generar_y_enviar_resumen_telegram():
    archivo_log = "registro_acciones.csv"
    if not os.path.exists(archivo_log):
        print("⚠ No hay acciones registradas aún.")
        return

    df = pd.read_csv(archivo_log)
    if df.empty:
        print("⚠ El archivo de registro está vacío.")
        return

    # --- Procesar datos
    resumen = df["Acción Tomada"].value_counts()
    rentabilidad = df.groupby("Acción Tomada")["Rentabilidad %"].mean()

    # --- Crear figura
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    # Pie chart
    axs[0].pie(resumen, labels=resumen.index, autopct="%1.1f%%", startangle=140)
    axs[0].set_title("Distribución de Decisiones")

    # Bar chart
    axs[1].bar(rentabilidad.index, rentabilidad.values, color="skyblue")
    axs[1].set_title("Rentabilidad Promedio")
    axs[1].set_ylabel("Rentabilidad %")
    axs[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    
    # Guardar imagen temporal
    nombre_archivo = f"resumen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(nombre_archivo)
    plt.close()

    # --- Enviar por Telegram
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

    # Borrar imagen temporal (opcional)
    os.remove(nombre_archivo)

def enviar_grafico_simulacion_telegram(fig, ticker):
    try:
        # Guardar la imagen del gráfico temporalmente
        nombre_archivo = f"simulacion_{ticker}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(nombre_archivo)
        plt.close(fig)

        # Enviar por Telegram
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

        os.remove(nombre_archivo)  # Limpiar imagen temporal
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

    # Enviar notificación por Telegram
    try:
        TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
        TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
        mensaje = f"📢 Acción registrada: *{accion}* para `{ticker}` con rentabilidad *{rentab:.2f}%*"
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        params = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": mensaje,
            "parse_mode": "Markdown"
        }
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

if archivo is not None:
    df = pd.read_excel(archivo, sheet_name="Inversiones")
    df.columns = df.columns.str.strip()

    if 'Ticker' in df.columns and 'Cantidad' in df.columns:
        df = df[df['Ticker'].notnull() & df['Cantidad'].notnull()]

         # Limpieza de columnas numéricas (con % y coma decimal)
        for col in ['Rentabilidad', 'Precio Actual', 'DCA']:
            if col in df.columns:
                temp = (
                    df[col]
                    .astype(str)
                    .str.replace("%", "", regex=False)  # Primero quita el %
                    .str.replace(",", ".", regex=False) # Luego cambia coma por punto
                )
                df[col] = pd.to_numeric(temp, errors="coerce")

        # Sección 1: Gestor
        if seccion == "Gestor de Portafolio":
            st.subheader("📊 Análisis de Posiciones")
            for _, row in df.iterrows():
                ticker = row["Ticker"]
                rentab = row["Rentabilidad"]
                precio = row["Precio Actual"]
                dca = row["DCA"]

                if pd.notna(rentab):
                    st.markdown(f"### ▶ {ticker}: {rentab:.2f}%")
                else:
                    st.markdown(f"### ▶ {ticker}: nan%")

                if pd.isna(rentab):
                    st.write("🔍 Revisión: Datos incompletos o mal formateados.")
                elif rentab >= 15:
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
                elif rentab > 8:
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

        # Sección 2: Simulador de Opciones
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
                        st.markdown(f"""
                - 🎯 Comprás el derecho a comprar la acción a ${strike_price} pagando una prima de ${premium:.2f}.
                - 📉 Si el precio final está **por debajo del strike**, **no ejercés** y pierdes solo la prima.
                - 📈 Si el precio sube **por encima de ${break_even:.2f}**, tienes ganancias netas.
                - ⚖️ El gráfico muestra tu rentabilidad según el precio al vencimiento.
                """)
                    elif rol == "Comprador" and tipo_opcion == "PUT":
                        st.markdown(f"""
                - 🎯 Comprás el derecho a vender la acción a ${strike_price} pagando una prima de ${premium:.2f}.
                - 📈 Ganás si la acción baja **por debajo de ${break_even:.2f}**.
                - 📉 Si se mantiene por encima del strike, la pérdida se limita a la prima.
                - ⚖️ El gráfico refleja tu cobertura o especulación a la baja.
                """)
                    elif rol == "Vendedor" and tipo_opcion == "CALL":
                        st.markdown(f"""
                - 💰 Vendés la opción y recibes ${premium:.2f} pero asumes la obligación de vender a ${strike_price}.
                - ✅ Si la acción cierra por debajo del strike, ganás toda la prima.
                - ⚠️ Si sube **por encima de ${break_even:.2f}**, comienzas a perder dinero.
                - 📉 Riesgo ilimitado si el precio sube mucho (a menos que tengas las acciones).
                """)
                    elif rol == "Vendedor" and tipo_opcion == "PUT":
                        st.markdown(f"""
                - 💰 Vendés la opción y te pagan ${premium:.2f} por asumir la obligación de comprar a ${strike_price}.
                - ✅ Ganás la prima si el precio se mantiene por encima del strike.
                - ⚠️ Si cae **por debajo de ${break_even:.2f}**, comienzas a perder dinero.
                - 📉 Riesgo limitado: como máximo hasta que la acción llegue a $0.
                """)

                with st.expander("📘 Perfil del rol seleccionado"):
                    if rol == "Comprador":
                        st.markdown(f"""
                - 💸 Pagás una prima (${premium:.2f}) por el derecho a ejercer.
                - 📈 Ganancia potencial ilimitada (CALL) o limitada (PUT).
                - 🔻 Pérdida máxima: la prima.
                """)
                    else:
                        if tipo_opcion == "CALL":
                            st.markdown(f"""
                - 💵 Recibes una prima (${premium:.2f}) por asumir la obligación de vender a ${strike_price}.
                - ✅ Ganancia máxima: la prima si la acción no supera el strike.
                - ⚠️ Si el precio sube por encima de ${break_even:.2f}, comienzas a tener pérdidas. Estas son potencialmente ilimitadas.
                - 🔒 Estrategia útil para generar ingresos si creés que la acción no superará el strike.
                """)
                        else:  # PUT vendedor
                            st.markdown(f"""
                - 💵 Recibes una prima (${premium:.2f}) por asumir la obligación de comprar a ${strike_price}.
                - ✅ Ganancia máxima: la prima si la acción se mantiene por encima del strike.
                - ⚠️ Si la acción cae por debajo de ${break_even:.2f}, empienzas a tener pérdidas. El riesgo es alto, pero finito (hasta que la acción llegue a $0).
                - 🛡 Estrategia usada si estás dispuesto a comprar la acción más barata que hoy.
                """)

                if st.button("📤 Enviar esta simulación a Telegram"):
                    enviar_grafico_simulacion_telegram(fig, selected_ticker)

            else:
                st.warning("⚠ No se encontró cadena de opciones para este ticker.")


        # Sección 3: Dashboard
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
else:
    st.info("Subí el archivo Excel para empezar.")

# --- Envío automático del resumen diario por Telegram a las 23hs ---
# from datetime import datetime
# ahora = datetime.now()
# if ahora.hour == 23 and ahora.minute < 5:
#     generar_y_enviar_resumen_telegram()
#     st.toast("📤 Resumen diario enviado automáticamente.")





