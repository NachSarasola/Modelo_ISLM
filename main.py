import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="IS-LM Básico", layout="wide")
st.title("Modelo IS-LM Básico (Estático)")

# Aplicar un tema estético
sns.set_theme(style="darkgrid")

# =======================================
# PARÁMETROS DEL MODELO IS-LM (ESTÁTICO)
# =======================================
st.sidebar.header("Parámetros del Modelo")

# --- Mercado de Bienes (IS) ---
a = st.sidebar.number_input("Consumo autónomo (a)", value=100.0)
b = st.sidebar.number_input("Propensión marginal a consumir (b)", value=0.8)
T0 = st.sidebar.number_input("Impuestos fijos (T0)", value=50.0)
lam = st.sidebar.number_input("Sensibilidad de impuestos (λ)", value=0.2)
I0 = st.sidebar.number_input("Inversión autónoma (I0)", value=150.0)
h = st.sidebar.number_input("Sensibilidad de inversión (h)", value=20.0)
G_bar = st.sidebar.number_input("Gasto público (Ḡ)", value=200.0)

# --- Mercado de Dinero (LM) ---
M_o = st.sidebar.number_input("Demanda autónoma de dinero (Mₒ)", value=50.0)
M_bar = st.sidebar.number_input("Oferta de dinero (M̄)", value=1000.0)
k = st.sidebar.number_input("Sensibilidad demanda de dinero (Y) (k)", value=0.5)
mu = st.sidebar.number_input("Sensibilidad demanda de dinero (r) (μ)", value=10.0)

# ============================
# FUNCIONES DEL MODELO IS-LM
# ============================

def is_curve(Y, a, b, T0, lam, I0, h, G_bar):
    """
    Curva IS (mercado de bienes):
    Y = C(Yd) + I(r) + G
    -> Despejamos r en función de Y:
       r_IS = [a - b*T0 + I0 + G_bar - (1 - b*(1-lam)) * Y] / h
    """
    return (a - b*T0 + I0 + G_bar - (1 - b*(1 - lam)) * Y) / h

def lm_curve(Y, M_o, M_bar, k, mu):
    """
    Curva LM (mercado de dinero):
    M_bar = M_o + k*Y - mu*r
    -> Despejamos r en función de Y:
       r_LM = (M_o + k*Y - M_bar) / mu
    """
    return (M_o + k*Y - M_bar) / mu

def equilibrium(a, b, T0, lam, I0, h, G_bar, M_o, M_bar, k, mu):
    """
    Cálculo del punto de equilibrio (Y*, r*) resolviendo:
      r_IS(Y*) = r_LM(Y*)
    """
    # r_IS(Y) = [a - bT0 + I0 + G_bar - (1 - b(1-lam))Y]/h
    # r_LM(Y) = (M_o + kY - M_bar)/mu
    # Igualar r_IS = r_LM:
    # => (a - bT0 + I0 + G_bar - (1 - b(1-lam))Y)/h = (M_o + kY - M_bar)/mu
    # => Y*(1 - b(1-lam) + (h*k)/mu) = (a - bT0 + I0 + G_bar) + (h/mu)(M_bar - M_o)
    #
    num = (a - b*T0 + I0 + G_bar) + (h/mu)*(M_bar - M_o)
    den = (1 - b*(1 - lam)) + (h*k)/mu
    Y_star = num / den
    # r* a partir de la LM
    r_star = (M_o + k*Y_star - M_bar) / mu
    # Evitamos tasas negativas
    r_star = max(r_star, 0)
    return Y_star, r_star

# ===============================
# GRÁFICA Y LÓGICA DE LA APLICACIÓN
# ===============================

if st.button("Mostrar IS-LM"):
    # 1) Rango de Y para graficar
    Y_range = np.linspace(0, 2000, 300)

    # 2) Calcular Curvas
    rIS_vals = is_curve(Y_range, a, b, T0, lam, I0, h, G_bar)
    rLM_vals = lm_curve(Y_range, M_o, M_bar, k, mu)

    # 3) Hallar punto de equilibrio
    Y_eq, r_eq = equilibrium(a, b, T0, lam, I0, h, G_bar, M_o, M_bar, k, mu)

    # 4) Evitar valores negativos en las curvas (por sentido económico)
    rIS_vals = np.maximum(rIS_vals, 0)
    rLM_vals = np.maximum(rLM_vals, 0)

    # ================
    # Gráfico IS y LM
    # ================
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Y_range, rIS_vals, label="Curva IS", color="blue", lw=2)
    ax.plot(Y_range, rLM_vals, label="Curva LM", color="red", lw=2)
    ax.plot(Y_eq, r_eq, 'ko', markersize=7, label="Equilibrio\n(Y*, r*)")

    ax.set_xlabel("Producto (Y)")
    ax.set_ylabel("Tasa de interés (r)")
    ax.set_title("Curvas IS y LM (Estático)")
    ax.legend()
    st.pyplot(fig)

    # ===========================
    # Mostrar valores de Equilibrio
    # ===========================
    st.markdown(f"**Punto de Equilibrio:**")
    st.markdown(f"- Y* = {Y_eq:.2f}")
    st.markdown(f"- r* = {r_eq:.2f}")