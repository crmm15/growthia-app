
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import os
import requests  # Para enviar mensajes a Telegram

st.set_page_config(page_title="Agent GrowthIA M&M", layout="wide")
st.title("🧠 Plataforma Integral para Gestión y Simulación de Inversiones")

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

        for col in ['Rentabilidad', 'Precio Actual', 'DCA']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".").str.replace("%", ""), errors="coerce")

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

        # Sección 2: Simulador
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
                fila = tabla_opciones.loc[np.abs(tabla_opciones["strike"] - strike_price).idxmin()]
                premium = (fila["bid"] + fila["ask"]) / 2

                st.markdown(f"**Precio actual:** ${precio_actual:.2f}")
                st.markdown(f"**Strike simulado:** ${strike_price}")
                st.markdown(f"**Prima estimada:** ${premium:.2f}")
                st.markdown(f"**Vencimiento elegido:** {fecha_venc}")

                if "delta" in fila:
                    prob = abs(fila["delta"]) * 100
                    st.markdown(f"📊 **Probabilidad implícita de alcanzar el strike (Delta): ~{prob:.1f}%**")

                # Simular el payoff
                S = np.linspace(precio_actual * 0.6, precio_actual * 1.4, 100)
                payoff = calcular_payoff_call(S, strike_price, premium) if tipo_opcion == "CALL" else calcular_payoff_put(S, strike_price, premium)

                fig, ax = plt.subplots(figsize=(5, 3))  # Tamaño ajustado
                ax.plot(S, payoff, label="Payoff")
                ax.axhline(0, color="gray", linestyle="--")
                ax.axvline(strike_price, color="red", linestyle="--")
                ax.set_xlabel("Precio al vencimiento")
                ax.set_ylabel("Ganancia / Pérdida")
                ax.set_title(f"{tipo_opcion} - {selected_ticker} ({nivel_riesgo})")
                ax.legend()
                st.pyplot(fig)

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





