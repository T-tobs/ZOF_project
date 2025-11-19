import streamlit as st
import pandas as pd
import math

# ----------------------------------------------------
#  Utility: Build polynomial function
# ----------------------------------------------------
def build_polynomial(coeffs):
    def f(x):
        total = 0
        p = len(coeffs) - 1
        for c in coeffs:
            total += c * (x ** p)
            p -= 1
        return total
    return f


def derivative(f, h=1e-6):
    return lambda x: (f(x + h) - f(x - h)) / (2 * h)


# ----------------------------------------------------
# Numerical Methods (track iteration logs)
# ----------------------------------------------------

def bisection(f, a, b, tol, max_iter):
    logs = []

    if f(a) * f(b) >= 0:
        return None, None, None, [{"error": "f(a) and f(b) must have opposite signs"}]

    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fa, fb, fc = f(a), f(b), f(c)
        error = abs(b - a)

        logs.append({
            "Iteration": i,
            "a": a, "b": b, "c": c,
            "f(a)": fa, "f(b)": fb, "f(c)": fc,
            "Error": error
        })

        if error < tol or fc == 0:
            return c, error, i, logs

        if fa * fc < 0:
            b = c
        else:
            a = c

    return c, error, max_iter, logs


def regula_falsi(f, a, b, tol, max_iter):
    logs = []

    if f(a) * f(b) >= 0:
        return None, None, None, [{"error": "f(a) and f(b) must have opposite signs"}]

    for i in range(1, max_iter + 1):
        fa, fb = f(a), f(b)
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        error = abs(fc)

        logs.append({
            "Iteration": i,
            "a": a, "b": b, "c": c,
            "f(a)": fa, "f(b)": fb, "f(c)": fc,
            "Error": error
        })

        if error < tol:
            return c, error, i, logs

        if fa * fc < 0:
            b = c
        else:
            a = c

    return c, error, max_iter, logs


def secant(f, x0, x1, tol, max_iter):
    logs = []
    for i in range(1, max_iter + 1):
        f0, f1 = f(x0), f(x1)
        if f1 - f0 == 0:
            return None, None, None, [{"error": "Division by zero"}]

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        error = abs(x2 - x1)

        logs.append({
            "Iteration": i,
            "x0": x0, "x1": x1, "x2": x2,
            "f(x0)": f0, "f(x1)": f1, "f(x2)": f(x2),
            "Error": error
        })

        if error < tol:
            return x2, error, i, logs

        x0, x1 = x1, x2

    return x2, error, max_iter, logs


def newton_raphson(f, x0, tol, max_iter):
    df = derivative(f)
    logs = []

    for i in range(1, max_iter + 1):
        fx = f(x0)
        dfx = df(x0)

        if dfx == 0:
            return None, None, None, [{"error": "Zero derivative"}]

        x1 = x0 - fx / dfx
        error = abs(x1 - x0)

        logs.append({
            "Iteration": i,
            "x0": x0,
            "f(x0)": fx,
            "df(x0)": dfx,
            "x1": x1,
            "Error": error
        })

        if error < tol:
            return x1, error, i, logs

        x0 = x1

    return x1, error, max_iter, logs


def fixed_point_iteration(g, x0, tol, max_iter):
    logs = []

    for i in range(1, max_iter + 1):
        x1 = g(x0)
        error = abs(x1 - x0)

        logs.append({
            "Iteration": i,
            "x0": x0,
            "x1": x1,
            "Error": error
        })

        if error < tol:
            return x1, error, i, logs

        x0 = x1

    return x1, error, max_iter, logs


def modified_secant(f, x0, delta, tol, max_iter):
    logs = []
    for i in range(1, max_iter + 1):
        fx = f(x0)
        d_approx = (f(x0 + delta * x0) - fx) / (delta * x0)

        if d_approx == 0:
            return None, None, None, [{"error": "Zero derivative approximation"}]

        x1 = x0 - fx / d_approx
        error = abs(x1 - x0)

        logs.append({
            "Iteration": i,
            "x0": x0,
            "f(x0)": fx,
            "x1": x1,
            "Error": error
        })

        if error < tol:
            return x1, error, i, logs

        x0 = x1

    return x1, error, max_iter, logs


# ----------------------------------------------------
# Streamlit GUI
# ----------------------------------------------------

st.title("🔢 ZOF Numerical Methods – Web GUI")

method = st.selectbox(
    "Choose a Root-Finding Method",
    [
        "Bisection Method",
        "Regula Falsi Method",
        "Secant Method",
        "Newton–Raphson Method",
        "Fixed Point Iteration",
        "Modified Secant Method"
    ]
)

st.subheader("Enter Polynomial Coefficients")
coeffs = st.text_input("Coefficients (highest degree first)", "1 0 -4")
coeffs = list(map(float, coeffs.split()))

tol = st.number_input("Tolerance", value=0.0001, step=0.0001)
max_iter = st.number_input("Maximum Iterations", value=50, step=1)

f = build_polynomial(coeffs)

# Method-specific parameters
a = b = x0 = x1 = delta = None
g = None

if method in ["Bisection Method", "Regula Falsi Method"]:
    a = st.number_input("a (interval start)", value=0.0)
    b = st.number_input("b (interval end)", value=5.0)

if method in ["Secant Method"]:
    x0 = st.number_input("x0", value=1.0)
    x1 = st.number_input("x1", value=3.0)

if method in ["Newton–Raphson Method", "Fixed Point Iteration", "Modified Secant Method"]:
    x0 = st.number_input("Initial guess x0", value=2.0)

if method == "Modified Secant Method":
    delta = st.number_input("Delta", value=0.01)

if method == "Fixed Point Iteration":
    st.info("Using g(x) = x - f(x)")
    g = lambda x: x - f(x)


# Run button
if st.button("Compute Root"):
    if method == "Bisection Method":
        root, err, iters, logs = bisection(f, a, b, tol, max_iter)

    elif method == "Regula Falsi Method":
        root, err, iters, logs = regula_falsi(f, a, b, tol, max_iter)

    elif method == "Secant Method":
        root, err, iters, logs = secant(f, x0, x1, tol, max_iter)

    elif method == "Newton–Raphson Method":
        root, err, iters, logs = newton_raphson(f, x0, tol, max_iter)

    elif method == "Fixed Point Iteration":
        root, err, iters, logs = fixed_point_iteration(g, x0, tol, max_iter)

    elif method == "Modified Secant Method":
        root, err, iters, logs = modified_secant(f, x0, delta, tol, max_iter)

    st.subheader("Iteration Details")
    df_logs = pd.DataFrame(logs)
    st.dataframe(df_logs)

    st.subheader("Final Result")
    st.write(f"**Estimated Root:** {root}")
    st.write(f"**Final Error:** {err}")
    st.write(f"**Iterations:** {iters}")
