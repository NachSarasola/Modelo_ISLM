import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import solve_ivp

# --- Definición del sistema de ODEs (dinámica) según las ecuaciones (11) y (12) ---
# dY/dt = α*(a - b*T0 + I0 + G_bar) - α*((1-b)*(1-λ)*Y) - α*h*r
# dr/dt = β*(M_o - M_bar + k*Y - μ*r)
def is_lm_odes(t, state, params):
    Y, r = state
    a      = params["a"]
    b      = params["b"]
    T0     = params["T0"]
    lam    = params["lambda"]
    I0     = params["I0"]
    h      = params["h"]
    G_bar  = params["G_bar"]
    alpha  = params["alpha"]
    M_o    = params["M_o"]
    M_bar  = params["M_bar"]
    k      = params["k"]
    mu     = params["mu"]
    beta   = params["beta"]

    dY_dt = alpha * (a - b*T0 + I0 + G_bar) - alpha * ((1 - b)*(1 - lam)*Y) - alpha * h * r
    dr_dt = beta * (M_o - M_bar + k*Y - mu*r)
    return [dY_dt, dr_dt]

# Función que calcula las curvas estáticas IS y LM
def static_curves(Y_range, params):
    a     = params["a"]
    b     = params["b"]
    T0    = params["T0"]
    lam   = params["lambda"]
    I0    = params["I0"]
    h     = params["h"]
    G_bar = params["G_bar"]
    M_o   = params["M_o"]
    M_bar = params["M_bar"]
    k     = params["k"]
    mu    = params["mu"]

    # Curva IS:
    # r_IS = [a - b*T0 + I0 + G_bar - Y*(1 - b*(1-lam))] / h
    r_IS = (a - b*T0 + I0 + G_bar - (1 - b*(1-lam)) * Y_range) / h

    # Curva LM:
    # r_LM = (M_o + k*Y - M_bar) / μ
    r_LM = (M_o + k*Y_range - M_bar) / mu

    return r_IS, r_LM

# Función para calcular el punto de equilibrio (intersección de IS y LM)
def equilibrium(params):
    a     = params["a"]
    b     = params["b"]
    T0    = params["T0"]
    lam   = params["lambda"]
    I0    = params["I0"]
    h     = params["h"]
    G_bar = params["G_bar"]
    M_o   = params["M_o"]
    M_bar = params["M_bar"]
    k     = params["k"]
    mu    = params["mu"]

    # Derivación:
    # (a - bT0 + I0 + G_bar - Y*(1 - b*(1-lam)))/h = (M_o + kY - M_bar)/mu
    # => Y*(1 - b*(1-lam) + (h*k)/mu) = a - bT0 + I0 + G_bar + (h/μ)*(M_bar - M_o)
    Y_eq = (a - b*T0 + I0 + G_bar + (h/mu)*(M_bar - M_o)) / (1 - b*(1-lam) + (h*k)/mu)
    # Usamos la ecuación LM para hallar r_eq
    r_eq = (M_o + k*Y_eq - M_bar) / mu
    return Y_eq, r_eq

# Función que ejecuta la simulación y actualiza ambos gráficos (dinámico y estático)
def simulate():
    try:
        # Extraer parámetros de las entradas de la interfaz
        params = {
            "a":      float(entries["a"].get()),
            "b":      float(entries["b"].get()),
            "T0":     float(entries["T0"].get()),
            "lambda": float(entries["lambda"].get()),
            "I0":     float(entries["I0"].get()),
            "h":      float(entries["h"].get()),
            "G_bar":  float(entries["G_bar"].get()),
            "alpha":  float(entries["alpha"].get()),
            "M_o":    float(entries["M_o"].get()),
            "M_bar":  float(entries["M_bar"].get()),
            "k":      float(entries["k"].get()),
            "mu":     float(entries["mu"].get()),
            "beta":   float(entries["beta"].get()),
        }
        # Condiciones iniciales para la parte dinámica
        Y0 = float(entries["Y0"].get())
        r0 = float(entries["r0"].get())
        # Tiempo de simulación
        t_start = 0
        t_end = float(entries["t_end"].get())
        t_span = (t_start, t_end)
        t_eval = np.linspace(t_start, t_end, 500)

        # Resolver el sistema dinámico de ODEs
        sol = solve_ivp(fun=lambda t, y: is_lm_odes(t, y, params),
                        t_span=t_span,
                        y0=[Y0, r0],
                        t_eval=t_eval)
        t_vals = sol.t
        Y_vals = sol.y[0]
        r_vals = sol.y[1]

        # --- Actualizar gráfico dinámico ---
        ax_time.clear()
        ax_phase.clear()

        # Gráfico: Evolución de Y y r vs. tiempo
        ax_time.plot(t_vals, Y_vals, label="Producto (Y)", color="blue")
        ax_time.plot(t_vals, r_vals, label="Tasa de interés (r)", color="red")
        ax_time.set_xlabel("Tiempo")
        ax_time.set_ylabel("Valores")
        ax_time.set_title("Evolución en el Tiempo (Dinámica)")
        ax_time.legend()

        # Diagrama de fase: r vs. Y
        ax_phase.plot(Y_vals, r_vals, color="green")
        ax_phase.set_xlabel("Producto (Y)")
        ax_phase.set_ylabel("Tasa de interés (r)")
        ax_phase.set_title("Diagrama de Fase (Dinámica)")

        canvas_dynamic.draw()

        # --- Actualizar gráfico estático (curvas IS y LM) ---
        Y_range = np.linspace(0, 2000, 500)
        r_IS, r_LM = static_curves(Y_range, params)
        Y_eq, r_eq = equilibrium(params)

        ax_static.clear()
        ax_static.plot(Y_range, r_IS, label="Curva IS", color="blue")
        ax_static.plot(Y_range, r_LM, label="Curva LM", color="red")
        ax_static.plot(Y_eq, r_eq, 'ko', label="Equilibrio\n(Y*, r*)")
        ax_static.set_xlabel("Producto (Y)")
        ax_static.set_ylabel("Tasa de interés (r)")
        ax_static.set_title("Diagrama Estático: Curvas IS y LM")
        ax_static.legend()

        canvas_static.draw()

    except Exception as e:
        print("Error en la simulación:", e)

# --- Configuración de la ventana principal ---
root = tk.Tk()
root.title("Simulación Dinámica y Estática IS-LM")

# Marco para parámetros de entrada
frame_params = ttk.Frame(root, padding="10")
frame_params.grid(row=0, column=0, sticky="NW")

# Lista de parámetros y etiquetas (con valores por defecto basados en la teoría)
param_labels = {
    "a": "Consumo autónomo (a):",
    "b": "Propensión marginal a consumir (b):",
    "T0": "Impuestos fijos (T0):",
    "lambda": "Sensibilidad de impuestos (λ):",
    "I0": "Inversión autónoma (I0):",
    "h": "Sensibilidad de inversión (h):",
    "G_bar": "Gasto público fijo (G_bar):",
    "alpha": "Velocidad de ajuste bienes (α):",
    "M_o": "Demanda autónoma de dinero (M_o):",
    "M_bar": "Oferta de dinero fija (M_bar):",
    "k": "Sensibilidad demanda de dinero (Y) (k):",
    "mu": "Sensibilidad demanda de dinero (r) (μ):",
    "beta": "Velocidad de ajuste dinero (β):",
    "Y0": "Producto inicial (Y0):",
    "r0": "Tasa de interés inicial (r0):",
    "t_end": "Tiempo de simulación (t_end):"
}

default_values = {
    "a": 100,
    "b": 0.8,
    "T0": 50,
    "lambda": 0.2,
    "I0": 150,
    "h": 20,
    "G_bar": 200,
    "alpha": 0.05,
    "M_o": 50,
    "M_bar": 1000,
    "k": 0.5,
    "mu": 10,
    "beta": 0.1,
    "Y0": 1000,
    "r0": 5,
    "t_end": 100
}

entries = {}
row = 0
for key, label_text in param_labels.items():
    ttk.Label(frame_params, text=label_text).grid(row=row, column=0, sticky="W")
    entry = ttk.Entry(frame_params)
    entry.grid(row=row, column=1, sticky="W")
    entry.insert(0, str(default_values[key]))
    entries[key] = entry
    row += 1

# Botón para ejecutar la simulación
button_simulate = ttk.Button(frame_params, text="Simular", command=simulate)
button_simulate.grid(row=row, column=0, columnspan=2, pady=10)

# --- Marco para gráficos dinámicos ---
frame_dynamic = ttk.Frame(root)
frame_dynamic.grid(row=0, column=1, padx=10, pady=10)

fig_dynamic, (ax_time, ax_phase) = plt.subplots(2, 1, figsize=(7, 8))
fig_dynamic.tight_layout(pad=3)
canvas_dynamic = FigureCanvasTkAgg(fig_dynamic, master=frame_dynamic)
canvas_dynamic.get_tk_widget().pack()

# --- Marco para gráfico estático (curvas IS y LM) ---
frame_static = ttk.Frame(root)
frame_static.grid(row=1, column=1, padx=10, pady=10)

fig_static, ax_static = plt.subplots(figsize=(7, 5))
canvas_static = FigureCanvasTkAgg(fig_static, master=frame_static)
canvas_static.get_tk_widget().pack()

# Ejecutar la aplicación
root.mainloop()