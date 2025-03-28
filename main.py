import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp

# Configurar Streamlit (debe ser lo primero)
st.set_page_config(page_title="Modelo IS-LM", layout="wide")
st.title("Simulación del Modelo IS-LM (Dinámica y Estática)")

# Aplicar tema con Seaborn (esto configura el estilo sin usar plt.style.use)
sns.set_theme(style="darkgrid")

# ==============================
# SECCIÓN: Parámetros del Modelo
# ==============================
st.sidebar.header("Parámetros del Modelo")

# Parámetros del mercado de bienes y fiscal
a = st.sidebar.number_input("Consumo autónomo (a)", value=100.0)
b = st.sidebar.number_input("Propensión marginal a consumir (b)", value=0.8)
T0 = st.sidebar.number_input("Impuestos fijos (T0)", value=50.0)
lam = st.sidebar.number_input("Sensibilidad de impuestos (λ)", value=0.2)
I0 = st.sidebar.number_input("Inversión autónoma (I0)", value=150.0)
h = st.sidebar.number_input("Sensibilidad de inversión (h)", value=20.0)
G_bar = st.sidebar.number_input("Gasto público fijo (Ḡ)", value=200.0)
alpha = st.sidebar.number_input("Velocidad de ajuste bienes (α)", value=0.05)

# Parámetros del mercado monetario
M_o = st.sidebar.number_input("Demanda autónoma de dinero (Mₒ)", value=50.0)
M_bar = st.sidebar.number_input("Oferta de dinero fija (M̄)", value=1000.0)
k = st.sidebar.number_input("Sensibilidad demanda de dinero (Y) (k)", value=0.5)
mu = st.sidebar.number_input("Sensibilidad demanda de dinero (r) (μ)", value=10.0)
beta = st.sidebar.number_input("Velocidad de ajuste dinero (β)", value=0.1)

# Condiciones iniciales y tiempo para la simulación dinámica
Y0 = st.sidebar.number_input("Producto inicial (Y₀)", value=1000.0)
r0 = st.sidebar.number_input("Tasa de interés inicial (r₀)", value=5.0)
# Se limita el tiempo máximo a 10
t_end = st.sidebar.number_input("Tiempo de simulación (máx 10)", value=10.0, max_value=10.0)

# Organizar parámetros en un diccionario (útil para las funciones)
params = {
    "a": a,
    "b": b,
    "T0": T0,
    "lambda": lam,
    "I0": I0,
    "h": h,
    "G_bar": G_bar,
    "alpha": alpha,
    "M_o": M_o,
    "M_bar": M_bar,
    "k": k,
    "mu": mu,
    "beta": beta,
}

# ==============================
# SECCIÓN: Definición del Modelo
# ==============================

def is_lm_odes(t, state, params):
    """
    Sistema de ODEs:
      dY/dt = α*(a - b*T0 + I0 + Ḡ) - α*((1-b)*(1-λ)*Y) - α*h*r
      dr/dt = β*(Mₒ - M̄ + k*Y - μ*r)
    """
    Y, r = state
    a = params["a"]
    b = params["b"]
    T0 = params["T0"]
    lam = params["lambda"]
    I0 = params["I0"]
    h = params["h"]
    G_bar = params["G_bar"]
    alpha = params["alpha"]
    M_o = params["M_o"]
    M_bar = params["M_bar"]
    k = params["k"]
    mu = params["mu"]
    beta = params["beta"]

    dY_dt = alpha * (a - b * T0 + I0 + G_bar) - alpha * ((1 - b) * (1 - lam) * Y) - alpha * h * r
    dr_dt = beta * (M_o - M_bar + k * Y - mu * r)
    return [dY_dt, dr_dt]

def static_curves(Y_range, params):
    """
    Calcula las curvas estáticas IS y LM:
      Curva IS:
        r_IS = [a - b*T0 + I0 + Ḡ - Y*(1 - b*(1-λ))] / h
      Curva LM:
        r_LM = (Mₒ + k*Y - M̄) / μ
    """
    a = params["a"]
    b = params["b"]
    T0 = params["T0"]
    lam = params["lambda"]
    I0 = params["I0"]
    h = params["h"]
    G_bar = params["G_bar"]
    M_o = params["M_o"]
    M_bar = params["M_bar"]
    k = params["k"]
    mu = params["mu"]

    r_IS = (a - b * T0 + I0 + G_bar - (1 - b * (1 - lam)) * Y_range) / h
    r_LM = (M_o + k * Y_range - M_bar) / mu
    # Evitamos tasas negativas: si alguna tasa es menor a 0, se recorta a 0.
    r_IS = np.maximum(r_IS, 0)
    r_LM = np.maximum(r_LM, 0)
    return r_IS, r_LM

def equilibrium(params):
    """
    Calcula el punto de equilibrio (intersección de las curvas IS y LM):

    Se iguala:
      [a - b*T0 + I0 + Ḡ - Y*(1 - b*(1-λ))]/h = (Mₒ + k*Y - M̄)/μ

    De donde:
      Y* = [a - b*T0 + I0 + Ḡ + (h/μ)(M̄ - Mₒ)] / [1 - b*(1-λ) + (h*k)/μ]
      r* se obtiene usando la ecuación LM.
    """
    a = params["a"]
    b = params["b"]
    T0 = params["T0"]
    lam = params["lambda"]
    I0 = params["I0"]
    h = params["h"]
    G_bar = params["G_bar"]
    M_o = params["M_o"]
    M_bar = params["M_bar"]
    k = params["k"]
    mu = params["mu"]

    Y_eq = (a - b * T0 + I0 + G_bar + (h / mu) * (M_bar - M_o)) / (1 - b * (1 - lam) + (h * k) / mu)
    r_eq = (M_o + k * Y_eq - M_bar) / mu
    # Recortar r_eq a 0 si es negativo
    r_eq = max(r_eq, 0)
    return Y_eq, r_eq

# ==============================
# SECCIÓN: Simulación y Gráficas
# ==============================

if st.button("Simular Modelo"):
    # --- Simulación Dinámica ---
    t_start = 0
    t_end_val = t_end  # Se asume que t_end <= 10
    t_span = (t_start, t_end_val)
    t_eval = np.linspace(t_start, t_end_val, 500)

    sol = solve_ivp(fun=lambda t, y: is_lm_odes(t, y, params),
                    t_span=t_span,
                    y0=[Y0, r0],
                    t_eval=t_eval)
    t_vals = sol.t
    Y_vals = sol.y[0]
    # Evitamos tasas negativas en la solución
    r_vals = np.maximum(sol.y[1], 0)

    # Gráfico: Evolución en el tiempo (tamaño compacto)
    fig_time, ax_time = plt.subplots(figsize=(5, 3))
    ax_time.plot(t_vals, Y_vals, label="Producto (Y)", color="blue", lw=2)
    ax_time.plot(t_vals, r_vals, label="Tasa de interés (r)", color="red", lw=2)
    ax_time.set_xlabel("Tiempo")
    ax_time.set_ylabel("Valor")
    ax_time.set_title("Evolución Dinámica")
    ax_time.legend(fontsize=9)
    ax_time.grid(True)
    st.pyplot(fig_time)

    # Diagrama de fase: r vs. Y (más compacto)
    fig_phase, ax_phase = plt.subplots(figsize=(5, 3))
    ax_phase.plot(Y_vals, r_vals, color="green", lw=2)
    ax_phase.set_xlabel("Producto (Y)")
    ax_phase.set_ylabel("Tasa de interés (r)")
    ax_phase.set_title("Diagrama de Fase")
    ax_phase.grid(True)
    st.pyplot(fig_phase)

    # --- Gráfica Estática (Curvas IS y LM) ---
    Y_range = np.linspace(0, 2000, 500)
    r_IS, r_LM = static_curves(Y_range, params)
    Y_eq, r_eq = equilibrium(params)

    fig_static, ax_static = plt.subplots(figsize=(5, 3))
    ax_static.plot(Y_range, r_IS, label="Curva IS", color="blue", lw=2)
    ax_static.plot(Y_range, r_LM, label="Curva LM", color="red", lw=2)
    ax_static.plot(Y_eq, r_eq, 'ko', markersize=6, label="Equilibrio\n(Y*, r*)")
    ax_static.set_xlabel("Producto (Y)")
    ax_static.set_ylabel("Tasa de interés (r)")
    ax_static.set_title("Curvas IS y LM (Estático)")
    ax_static.legend(fontsize=9)
    ax_static.grid(True)
    st.pyplot(fig_static)