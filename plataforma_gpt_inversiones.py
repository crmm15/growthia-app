
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import os

st.set_page_config(page_title="Agent GrowthIA M&M", layout="wide")
st.title("🧠 Plataforma Integral para Gestión y Simulación de Inversiones")

# Menú principal
seccion = st.sidebar.radio("📂 Elegí una sección", ["Inicio", "Gestor de Portafolio", "Simulador de Opciones", "Dashboard de Desempeño"])

def calcular_payoff_call(S, K, premium):
    return np.maximum(S - K, 0) - premium

def calcular_payoff_put(S, K, premium):
    return np.maximum(K - S, 0) - premium
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

        # Sección 2: Simulador
        elif seccion == "Simulador de Opciones":
            st.subheader("📈 Simulador de Opciones con Perfil de Riesgo")
            selected_ticker = st.selectbox("Seleccioná un ticker", df["Ticker"].unique())
            nivel_riesgo = st.radio(
                "🎯 Tu perfil de riesgo",
                ["Conservador", "Balanceado", "Agresivo"],
                index=1,
                help="Define la tolerancia al riesgo. Afecta cuánto margen al alza o baja se permite sobre el strike."
            )
            tipo_opcion = st.radio(
                "Tipo de opción",
                ["CALL", "PUT"],
                help="CALL = derecho a comprar. PUT = derecho a vender. Elige según tu visión de mercado."
            )
            sugerencia = {"Conservador": 5, "Balanceado": 10, "Agresivo": 20}
            delta_strike = st.slider(
                "🧮 % sobre el precio actual para el strike",
                -30, 30, sugerencia[nivel_riesgo],
                help="Ajustá el precio strike en relación al precio actual del activo. ±% define cuán in/out of the money está."
            )
            dias_a_vencimiento = st.slider(
                "📆 Días hasta vencimiento",
                7, 90, 30,
                help="Duración restante del contrato. Más días = más prima (valor temporal)."
            )

            datos = df[df["Ticker"] == selected_ticker].iloc[0]
            precio_actual = datos["Precio Actual"]
            strike_price = round(precio_actual * (1 + delta_strike / 100), 2)

            ticker_yf = yf.Ticker(selected_ticker)
            expiraciones = ticker_yf.options
            if expiraciones:
                fecha_venc = min(expiraciones, key=lambda x: abs((pd.to_datetime(x) - pd.Timestamp.today()).days - dias_a_vencimiento))
                cadena = ticker_yf.option_chain(fecha_venc)
                tabla_opciones = cadena.calls if tipo_opcion == "CALL" else cadena.puts
                fila = tabla_opciones.loc[np.abs(tabla_opciones["strike"] - strike_price).idxmin()]
                premium = (fila["bid"] + fila["ask"]) / 2

                st.markdown(f"Precio actual: ${precio_actual:.2f}")
                st.markdown(f"Strike simulado: ${strike_price}")
                st.markdown(f"Prima estimada: ${premium:.2f}")
                st.markdown(f"Vencimiento elegido: {fecha_venc}")
                if "delta" in fila:
                    probabilidad = abs(fila['delta']) * 100
                    st.markdown(
                        f"📊 **Probabilidad implícita de alcanzar el strike:** ~{probabilidad:.1f}%"
                    )
                    st.caption(
                        "ℹ️ Basado en el delta de la opción. Aproximación de que termine 'in-the-money' al vencimiento."
                )

                S = np.linspace(precio_actual * 0.6, precio_actual * 1.4, 100)
                payoff = calcular_payoff_call(S, strike_price, premium) if tipo_opcion == "CALL" else calcular_payoff_put(S, strike_price, premium)

                fig, ax = plt.subplots()
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
