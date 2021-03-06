import mpmath as mp
import cmath

class NativeVSPreciseError(Exception):
    """Difference between Native implementatino and Precise implementation exceeded threhold"""
    pass

class HerbieVSPreciseError(Exception):
    """Difference between Herbie implementatino and Precise implementation exceeded threhold"""
    pass

class NativeVSHerbieError(Exception):
    """Difference between Native implementatino and Herbie implementation exceeded threhold"""
    pass

# quadratic formulas
def solve_quadratic_Native(a, b, c):
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

    x_Native   = solve_quadratic_Native(a, b, c)
    x_herbie  = solve_quadratic_herbie(a, b, c)
    x_precise = solve_quadratic_precise(a, b, c)

    x_native_vs_precise_error  = measure_error(x_native,  x_precise)
    x_herbie_vs_precise_error = measure_error(x_herbie, x_precise)
    x_native_vs_herbie_error   = measure_error(x_native,  x_herbie)

    exceptions = []
    if x_Native_vs_precise_error > ERROR_THRESHOLD:
        print("Difference between Native and Precise precision error of %s using a=%s, b=%s, c=%s" % (x_native_vs_precise_error, a, b, c))
        exceptions.append(NativeVSPreciseError())    
    if x_herbie_vs_precise_error > ERROR_THRESHOLD:
        print("Difference between Herbie and Precise precision error of %s using a=%s, b=%s, c=%s" % (x_herbie_vs_precise_error, a, b, c))
        exceptions.append(HerbieVSPreciseError())
    if x_Native_vs_herbie_error > ERROR_THRESHOLD:
        print("Difference between Native and Herbie precision error of %s using a=%s, b=%s, c=%s" % (x_native_vs_herbie_error, a, b, c))
        exceptions.append(NativeVSHerbieError())

    if exceptions:
        raise Exception([exceptions])