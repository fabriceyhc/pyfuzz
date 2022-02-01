import mpmath as mp
import cmath

class NaiveError(Exception):
    """Naive precision error"""

class HerbieError(Exception):
    """Herbie precision error"""

# quadratic formulas
def solve_quadratic_naive(a, b, c):
    if a == 0:
        return (-1, -1)

    # calculate the discriminant
    d = (b**2) - (4*a*c)

    # find two solutions
    sol1 = (-b-cmath.sqrt(d))/(2*a)
    sol2 = (-b+cmath.sqrt(d))/(2*a)

    return (sol1, sol2)

def solve_quadratic_herbie(a, b, c):
    if a == 0:
        return (-1, -1)

    if b < 0:
        sol1 = ((4*a*c) / (-b + cmath.sqrt(b**2 - 4*a*c)))/(2*a)
        sol2 = ((4*a*c) / (-b - cmath.sqrt(b**2 - 4*a*c)))/(2*a)
    elif 0 <= b and b <= 1e127:
        sol1 = (-b - cmath.sqrt(b**2 - 4*a*c))/(2*a)
        sol2 = (-b + cmath.sqrt(b**2 - 4*a*c))/(2*a)
    else: # elif 1e127 < b
        sol1 = -(b/a) + (c/b)
        sol2 = 0

    return sol1, sol2

def solve_quadratic_precise(a, b, c, precision_level=2000):
    if a == 0:
        return (-1, -1)

    mp.prec = precision_level

    # calculate the discriminant
    d = (b**2) - (4*a*c)
    
    # find two solutions
    sol1 = (-b-mp.sqrt(d))/(2*a)
    sol2 = (-b+mp.sqrt(d))/(2*a)

    return (sol1, sol2)

def measure_error(values1, values2):

    values1_mpc = tuple(mp.mpc(x) for x in values1)
    values2_mpc = tuple(mp.mpc(x) for x in values2)

    err1 = abs(values1_mpc[0] - values2_mpc[0])
    err2 = abs(values1_mpc[1] - values2_mpc[1])
    
    err = err1 + err2
    
    return err

def quadratic_comparison(a, b, c, ERROR_THRESHOLD=0.0):

    x_naive   = solve_quadratic_naive(a, b, c)
    x_herbie  = solve_quadratic_herbie(a, b, c)
    x_precise = solve_quadratic_precise(a, b, c)

    # print(x_naive, x_herbie, x_precise)

    x_naive_error  = measure_error(x_naive,  x_precise)
    x_herbie_error = measure_error(x_herbie, x_precise)

    # print(x_naive_error, x_herbie_error)

    if x_naive_error > ERROR_THRESHOLD:
        print("Naive precision error of %s using a=%s, b=%s, c=%s" % (x_naive_error, a, b, c))
        raise NaiveError()
    if x_herbie_error > ERROR_THRESHOLD:
        print("Herbie precision error of %s using a=%s, b=%s, c=%s" % (x_herbie_error, a, b, c))
        raise HerbieError()
