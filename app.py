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
# Streamlit GUI (Improved)
# ----------------------------------------------------

st.set_page_config(page_title="ZOF Numerical Methods", layout="centered")

st.markdown("<h1 style='text-align:center;'>🔢 CSC 431 Root Finding Methods – Web GUI</h1>",
            unsafe_allow_html=True)

st.write("Compute roots of equations using classical numerical methods.\n"
         "Enter parameters, run the solver, and view iteration results.")

st.markdown("---")

# Select method
method = st.selectbox(
    "Select a Root-Finding Method:",
    [
        "Bisection Method",
        "Regula Falsi Method",
        "Secant Method",
        "Newton–Raphson Method",
        "Fixed Point Iteration",
        "Modified Secant Method"
    ]
)

# Polynomial input
st.subheader("📘 Polynomial Function")
coeffs = st.text_input("Enter polynomial coefficients (highest degree first)", "1 0 -4")
coeffs = list(map(float, coeffs.split()))

tol_col, iter_col = st.columns(2)
tol = tol_col.number_input("Tolerance", value=0.0001, step=0.0001, format="%.6f")
max_iter = iter_col.number_input("Maximum Iterations", value=50, step=1)

f = build_polynomial(coeffs)

st.markdown("---")

# Method-specific inputs
st.subheader("⚙ Method Parameters")

if method in ["Bisection Method", "Regula Falsi Method"]:
    colA, colB = st.columns(2)
    a = colA.number_input("Interval start (a)", value=0.0)
    b = colB.number_input("Interval end (b)", value=5.0)

if method == "Secant Method":
    colA, colB = st.columns(2)
    x0 = colA.number_input("Initial guess x0", value=1.0)
    x1 = colB.number_input("Initial guess x1", value=3.0)

if method in ["Newton–Raphson Method", "Fixed Point Iteration", "Modified Secant Method"]:
    x0 = st.number_input("Initial guess x0", value=2.0)

if method == "Modified Secant Method":
    delta = st.number_input("Delta value", value=0.01)

if method == "Fixed Point Iteration":
    st.info("Using iteration function  **g(x) = x - f(x)**")
    g = lambda x: x - f(x)
else:
    g = None

st.markdown("---")

# Compute button
center_btn = st.columns([3, 2, 3])
with center_btn[1]:
    run_clicked = st.button("🚀 Compute Root", use_container_width=True)

# Run solver
if run_clicked:
    try:
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

        # Handle invalid cases
        if isinstance(logs, list) and "error" in logs[0]:
            st.error(logs[0]["error"])
            st.stop()

        # Show iteration table
        st.subheader("📊 Iteration Details")
        df_logs = pd.DataFrame(logs)
        st.dataframe(df_logs, use_container_width=True, height=300)

        # Show final results
        st.subheader("🏁 Final Result")
        st.success(
            f"""
            **Estimated Root:** `{root}`  
            **Final Error:** `{err}`  
            **Iterations:** `{iters}`
            """
        )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
