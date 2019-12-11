import pylab
import copy
from sympy import diff, symbols, cos, sin
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from sympy.solvers.solveset import linsolve
from scipy.optimize import rosen_der, rosen_hess

print_Hesse_matrix = True
print_Jakobi_matrix = True
step = 0.01
x_l = -2.0
y_l = -4.0


def graphics(x, y, z, point=None, with_point=None):
    fig = pylab.figure()
    axes = Axes3D(fig)
    axes.plot_surface(x, y, z, color="yellow")
    global step
    global x_l
    global y_l

    linear_r_s = [-1, -1]
    non_linear_r_s = [-2]
    new_x = copy.deepcopy(x)
    new_y = copy.deepcopy(y)
    new_z = copy.deepcopy(z)
    for i_index, x_item in enumerate(new_x[0]):
        for j_index, y_item in enumerate(new_y):
            if (g1([x_item, y_item[0]]) > linear_r_s[0]) or (g2([x_item, y_item[0]]) > linear_r_s[1]) \
                    or (g3([x_item, y_item[0]]) > non_linear_r_s[0]):
                new_z[int((y_item[0] - y_l) / step)][int((x_item - x_l) / step)] = np.nan
    axes.plot_surface(new_x, new_y, new_z, color="purple")
    if with_point:
        axes.scatter(point[0], point[1], func([point[0], point[1]]), color='red', s=40, marker='o')

    pylab.show()


def func(point):
    x, y = point
    return (2*y/3 - x - 2) + 10 * (3*y - x**3)**2


def g1(point):
    x, y = point
    return y - 2*x


def g2(point):
    x, y = point
    return y + 2*x


def g3(point):
    x, y = point
    return -10 * (x + 1) ** 2 - (y - 2) ** 2


def cons_f(x):
    return [-10 * (x[0] + 1) ** 2 - (x[1] - 2) ** 2]


def cons_J(x):
    global print_Jakobi_matrix
    x_, y_ = symbols('x y')
    non_lin_cons = cons_f([x_, y_])
    Jakobi = []

    for inequality in non_lin_cons:
        df_x_y = []
        df_x = inequality.diff(x_)
        df_y = inequality.diff(y_)
        df_x_y.append(df_x)
        df_x_y.append(df_y)
        Jakobi.append(df_x_y)
    if print_Jakobi_matrix:
        print("\n")
        print("Jakobi matrix: ")
        print(Jakobi)
        print("\n")
        print_Jakobi_matrix = False
    results = []
    for Jakobi_item in Jakobi:
        tmp = []
        for item in Jakobi_item:
            tmp.append(float(item.subs({x_: x[0], y_: x[1]})))
        results.append(tmp)
    return results


def cons_H(x, v):
    global print_Hesse_matrix
    x_, y_ = symbols('x y')
    non_lin_cons = cons_f([x_, y_])
    Hesses = []

    for inequality in non_lin_cons:
        df_xx = inequality.diff(x_).diff(x_)
        df_yy = inequality.diff(y_).diff(y_)
        df_xy = inequality.diff(x_).diff(y_)
        df_yx = inequality.diff(y_).diff(x_)
        Hesses.append([[df_xx, df_xy], [df_yx, df_yy]])

    if print_Hesse_matrix:
        print("Hesse matrix: ")
        print(Hesses)
        print("\n")
        print_Hesse_matrix = False
    results = []
    for Hesse in Hesses:
        calc_line = []
        for line in Hesse:
            tmp = []
            for item in line:
                tmp.append(float(item.subs({x_: x[0], y_: x[1]})))
            calc_line.append(tmp)
        results.append(calc_line)
    return v[0] * np.array(results[0])

def get_lambda(x_input):
    r_s_l = [-1, -1]
    r_s_nl = [-20]
    eps = 0.01
    active_intersection = []
    x, y = symbols('x y')
    g = [g1([x, y]), g2([x, y]), g3([x, y])]
    linear_r_s = [-1, -1]
    non_linear_r_s = [-20]
    print("\n")
    print("All constraints")
    print(g[0], " <= ", r_s_l[0])
    print(g[1], " <= ", r_s_l[1])
    print(g[2], " <= ", r_s_nl[0])
    print("\n")
    if -eps <= g[0].subs({x: x_input[0], y: x_input[1]}) - linear_r_s[0] <= eps:
        active_intersection.append(g[0])
    if -eps <= g[1].subs({x: x_input[0], y: x_input[1]}) - linear_r_s[1] <= eps:
        active_intersection.append(g[1])
    if -eps <= g[2].subs({x: x_input[0], y: x_input[1]}) - non_linear_r_s[0] <= eps:
        active_intersection.append(g[2])
    print("The number of active intersection is : ", len(active_intersection))
    lambda_ = [symbols('l1'), symbols('l2')]

    L = func([x, y])
    for index, inters in enumerate(active_intersection):
        L = L + lambda_[index] * active_intersection[index]
    print("Lagrange function")
    print(L)
    d_f_x = L.diff(x)
    d_f_y = L.diff(y)
    d_f_x_in_point = d_f_x.subs({x: x_input[0], y: x_input[1]})
    d_f_y_in_point = d_f_y.subs({x: x_input[0], y: x_input[1]})
    print("\n")
    print("System of equations in point:")
    print("df_x : ", d_f_x_in_point)
    print("df_y : ", d_f_y_in_point)

    print("LAMBDA: \n")
    solve = linsolve([d_f_x_in_point, d_f_y_in_point], lambda_)
    print(solve)

    return solve


def main():
    x, y = np.meshgrid(np.arange(-2.0, 2.0, 0.01), np.arange(-4.0, 4.0, 0.01))
    z = func([x, y])
    graphics(x, y, z)
    non_linear_r_s = [20]

    x_i = 1.0
    y_i = 1.0

    matr = [[-1.0, 1.0], [1.0, 1.0]]
    r_s = [-1.0, -1.0]
    linear_constraint = LinearConstraint(matr, [-np.inf, -np.inf], r_s)
    nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, non_linear_r_s[0],
                                               jac=cons_J, hess=cons_H)
    x0 = np.array([x_i, y_i])
    res = minimize(func, x0, method='trust-constr', jac=rosen_der, hess=rosen_hess,
                   constraints=[linear_constraint, nonlinear_constraint], bounds=((float(-2),
                                                                                   float(2)),
                                                                                  (float(-4),
                                                                                   float(0))),
                   )
    print("MINIMUM: ")
    print(res.x)
    opt = res.x.reshape(-1, 1)
    get_lambda(res.x)
    graphics(x, y, z, point=opt, with_point=True)

if __name__ == "__main__":
    main()