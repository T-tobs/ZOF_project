import argparse
import math
import sys

# ----------------------------------------------------
#  Utility: parse polynomial coefficients into function
# ----------------------------------------------------
def build_polynomial(coeffs):
    """
    Given coefficients [a_n, ..., a_1, a_0], return a function f(x)
    """
    def f(x):
        total = 0
        power = len(coeffs) - 1
        for c in coeffs:
            total += c * (x ** power)
            power -= 1
        return total
    return f


def derivative(f, h=1e-6):
    """
    Numerical derivative for Newton-Raphson
    """
    return lambda x: (f(x+h) - f(x-h)) / (2*h)


# ----------------------------------------------------
#  Root-Finding Algorithms
# ----------------------------------------------------

def bisection(f, a, b, tol, max_iter):
    print("\n=== BISECTION METHOD ===")

    if f(a) * f(b) >= 0:
        print("Error: f(a) and f(b) must have opposite signs.")
        sys.exit()

    for i in range(1, max_iter+1):
        c = (a + b) / 2
        fa, fb, fc = f(a), f(b), f(c)
        error = abs(b - a)

        print(f"Iter {i}: a={a:.6f}, b={b:.6f}, c={c:.6f}, f(c)={fc:.6f}, error={error:.6f}")

        if error < tol or fc == 0:
            return c, error, i

        if fa * fc < 0:
            b = c
        else:
            a = c
    
    return c, error, max_iter


def regula_falsi(f, a, b, tol, max_iter):
    print("\n=== REGULA FALSI (FALSE POSITION) METHOD ===")

    if f(a) * f(b) >= 0:
        print("Error: f(a) and f(b) must have opposite signs.")
        sys.exit()

    for i in range(1, max_iter+1):
        fa, fb = f(a), f(b)
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        error = abs(fc)

        print(f"Iter {i}: a={a:.6f}, b={b:.6f}, c={c:.6f}, f(c)={fc:.6f}, error={error:.6f}")

        if error < tol:
            return c, error, i

        if fa * fc < 0:
            b = c
        else:
            a = c

    return c, error, max_iter


def secant(f, x0, x1, tol, max_iter):
    print("\n=== SECANT METHOD ===")

    for i in range(1, max_iter+1):
        f0, f1 = f(x0), f(x1)
        if f1 - f0 == 0:
            print("Error: division by zero in Secant method.")
            sys.exit()

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        error = abs(x2 - x1)

        print(f"Iter {i}: x0={x0:.6f}, x1={x1:.6f}, x2={x2:.6f}, f(x2)={f(x2):.6f}, error={error:.6f}")

        if error < tol:
            return x2, error, i

        x0, x1 = x1, x2

    return x2, error, max_iter


def newton_raphson(f, x0, tol, max_iter):
    print("\n=== NEWTON-RAPHSON METHOD ===")

    df = derivative(f)

    for i in range(1, max_iter+1):
        fx = f(x0)
        dfx = df(x0)

        if dfx == 0:
            print("Error: derivative is zero.")
            sys.exit()

        x1 = x0 - fx / dfx
        error = abs(x1 - x0)

        print(f"Iter {i}: x0={x0:.6f}, f(x0)={fx:.6f}, x1={x1:.6f}, error={error:.6f}")

        if error < tol:
            return x1, error, i

        x0 = x1

    return x1, error, max_iter


def fixed_point_iteration(g, x0, tol, max_iter):
    print("\n=== FIXED POINT ITERATION METHOD ===")

    for i in range(1, max_iter+1):
        x1 = g(x0)
        error = abs(x1 - x0)

        print(f"Iter {i}: x0={x0:.6f}, x1={x1:.6f}, error={error:.6f}")

        if error < tol:
            return x1, error, i

        x0 = x1

    return x1, error, max_iter


def modified_secant(f, x0, delta, tol, max_iter):
    print("\n=== MODIFIED SECANT METHOD ===")

    for i in range(1, max_iter+1):
        fx = f(x0)
        d_approx = (f(x0 + delta * x0) - fx) / (delta * x0)

        if d_approx == 0:
            print("Error: zero derivative approximation.")
            sys.exit()

        x1 = x0 - fx / d_approx
        error = abs(x1 - x0)

        print(f"Iter {i}: x0={x0:.6f}, x1={x1:.6f}, f(x0)={fx:.6f}, error={error:.6f}")

        if error < tol:
            return x1, error, i

        x0 = x1

    return x1, error, max_iter


# ----------------------------------------------------
#  CLI Argument Parsing
# ----------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ZOF_CLI: Root-Finding Methods.")

    parser.add_argument(
        "--method", type=str, required=True, choices=[
            "bisection", "regula", "secant", "newton", "fixed", "modified_secant"
        ], help="Select the root-finding method."
    )

    parser.add_argument("--coeffs", nargs="+", type=float, required=True,
                        help="Polynomial coefficients from highest degree to constant term.")

    parser.add_argument("--a", type=float, help="Left interval for bisection/regula.")
    parser.add_argument("--b", type=float, help="Right interval for bisection/regula.")

    parser.add_argument("--x0", type=float, help="Initial guess x0.")
    parser.add_argument("--x1", type=float, help="Second initial guess x1 (Secant).")

    parser.add_argument("--delta", type=float, help="Delta for modified secant.")

    parser.add_argument("--tol", type=float, required=True, help="Tolerance.")
    parser.add_argument("--max_iter", type=int, required=True, help="Max iterations.")

    args = parser.parse_args()

    f = build_polynomial(args.coeffs)

    if args.method == "bisection":
        root, err, iters = bisection(f, args.a, args.b, args.tol, args.max_iter)

    elif args.method == "regula":
        root, err, iters = regula_falsi(f, args.a, args.b, args.tol, args.max_iter)

    elif args.method == "secant":
        root, err, iters = secant(f, args.x0, args.x1, args.tol, args.max_iter)

    elif args.method == "newton":
        root, err, iters = newton_raphson(f, args.x0, args.tol, args.max_iter)

    elif args.method == "fixed":
        # user must provide a g(x) manually; for now assume g(x)=x - f(x)
        g = lambda x: x - f(x)
        root, err, iters = fixed_point_iteration(g, args.x0, args.tol, args.max_iter)

    elif args.method == "modified_secant":
        root, err, iters = modified_secant(f, args.x0, args.delta, args.tol, args.max_iter)

    print("\n=== RESULT ===")
    print(f"Estimated Root: {root}")
    print(f"Final Error: {err}")
    print(f"Iterations: {iters}")


if __name__ == "__main__":
    main()
