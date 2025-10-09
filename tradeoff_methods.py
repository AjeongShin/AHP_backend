import numpy as np
from scipy.optimize import linprog, minimize

# Function to get RI (Random Index) value
def get_random_index(n):
    ri_table = {
        1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
    }
    return ri_table.get(n, 1.49)

# Calculation function based on the Eigenvector method
def ahp_eigen_solver(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_index = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues.real[max_index]
    eigenvector = np.abs(eigenvectors[:, max_index].real)
    weights = eigenvector / np.sum(eigenvector)

    n = matrix.shape[0]
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0
    ri = get_random_index(n)
    cr = ci / ri if ri else 0

    return weights.tolist(), lambda_max, ci, cr

def linear_bwm_solver(n, criteria, best_idx, worst_idx, aB, aW, epsilon=1e-6):
    #########################
    # solve optimization prob(linear BWM) by using scipy.linprog 
    # return value would be same with AHP(wiehg, ci, cr ..)
    ########################
    updated_w, updated_xi = minimize_xi(n, best_idx, worst_idx, aB, aW, epsilon)
    low_weights = lower_bound_weights(n, best_idx, worst_idx, aB, aW, epsilon, updated_w, updated_xi)
    upper_weights = maximize_weights(n, best_idx, worst_idx, aB, aW, epsilon, updated_w, updated_xi)
    rank, ci, cr = calculate_rank(n, low_weights, upper_weights)
    weights = updated_w
    # lambda_max = None  # Not applicable in BWM      
    return weights, rank, ci, cr


def create_constraints(n, best_idx, worst_idx, aB, aW, epsilon=1e-6, fixed_xi=None, use_xi_variable=True):
    """
    Parameters:
    -----------
    fixed_xi : float, optional
        use fixed xi value (Step2, Step3)
    use_xi_variable : bool
        w vector includes xi value (Step1: True, Step2/3: False)
    """
    # equality constraint: sum of weights = 1 (f(x) = 0)
    def constraint_eq(w):
        if use_xi_variable:
            return np.sum(w[:-1]) - 1  # w = [w1,...,wn, xi]
        else:
            return np.sum(w) - 1  # w = [w1,...,wn]
    
    constraints = [{'type':'eq', 'fun': constraint_eq}]

    for j in range(n):
        # Inequality constraints:  h(w) >= 0
        def constraint_ineq_best_positive(j=j):
            def constraint(w):
                if use_xi_variable:
                    weights = w[:-1]
                    xi = w[-1]
                else:
                    weights = w
                    xi = fixed_xi
                weight_best = weights[best_idx]
                return xi - (weight_best / w[j] - aB[j])
            return constraint
            
        def constraint_ineq_best_negative(j=j):
            def constraint(w):
                if use_xi_variable:
                    weights = w[:-1]
                    xi = w[-1]
                else:
                    weights = w
                    xi = fixed_xi
                weight_best = weights[best_idx]
                return xi + (weight_best / w[j] - aB[j])
            return constraint

        
        def constraint_ineq_worst_positive(j=j):
            def constraint(w):
                if use_xi_variable:
                    weights = w[:-1]
                    xi = w[-1]
                else:
                    weights = w
                    xi = fixed_xi
                weight_worst = weights[worst_idx]
                return xi - (w[j] / weight_worst + aW[j])
            return constraint

        
        def constraint_ineq_worst_negative(j=j):
            def constraint(w):
                if use_xi_variable:
                    weights = w[:-1]
                    xi = w[-1]
                else:
                    weights = w
                    xi = fixed_xi
                weight_worst = weights[worst_idx]
                return xi + (w[j] / weight_worst - aW[j])
            return constraint
            
        # minimize requries constraints to be in the form of dictionaries
        constraints.append({'type': 'ineq', 'fun': constraint_ineq_best_positive()})
        constraints.append({'type': 'ineq', 'fun': constraint_ineq_best_negative()})
        constraints.append({'type': 'ineq', 'fun': constraint_ineq_worst_positive()})
        constraints.append({'type': 'ineq', 'fun': constraint_ineq_worst_negative()})

    return constraints

# ========================================
# Step 1: Minimize Xi
# ========================================
def minimize_xi(n, best_idx, worst_idx, aB, aW, epsilon=1e-6):
    # objective function: minimize xi
    # w = [w1, w2, ..., wn, xi]
    def obj(w):
        xi = w[-1]
        return xi
    
    # constraints of equality and inequality
    constraints = create_constraints(n, best_idx, worst_idx, aB, aW, epsilon,
        fixed_xi=None, use_xi_variable=True)
    # bounds for weights and xi
    bounds = [(epsilon, None) for _ in range(n)] + [(0, None)]  # weights between epsilon and 1, xi >= 0
    
    # initial guess with equal weights and small xi
    w0 = np.array([1/n] * n + [0.1])
    result = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=constraints,     
        options={
            'ftol': 1e-9,       
            'disp': False,      
            'maxiter': 1000     
        }
    )
    updated_w = result[:n].tolist()
    updated_xi = result[-1]

    return updated_w, updated_xi

# ========================================
# Step 2: Lower Bound Weights
# ========================================
def lower_bound_weights(n, best_idx, worst_idx, aB, aW, updated_w, updated_xi, epsilon=1e-6):
    low_weights = []

    # constraints of equality and inequality
    constraints = create_constraints(n, best_idx, worst_idx, aB, aW, epsilon, fixed_xi=updated_xi, use_xi_variable=False )

    # bounds for weights and xi
    bounds = [(epsilon, None) for _ in range(n)]  # weights between epsilon and 1
    
    # initial guess with equal weights and updated xi
    w0 = np.array(updated_w)
    
    for j in range(n):
        def obj(w, j=j):
            return w[j]
        
        result = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=constraints,     
            options={
                'ftol': 1e-9,       
                'disp': False,      
                'maxiter': 1000     
            }
        )

        min_weight = result[j]
        low_weights.append(min_weight)

    return low_weights

# ========================================
# Step 3: Upper Bound Weights
# ========================================
def maximize_weights(n, best_idx, worst_idx, aB, aW, updated_w, updated_xi, epsilon=1e-6):
    upper_weights = []

    # constraints of equality and inequality
    constraints = create_constraints(n, best_idx, worst_idx, aB, aW, epsilon, fixed_xi=updated_xi, use_xi_variable=False )

    # bounds for weights and xi
    bounds = [(epsilon, None) for _ in range(n)]  # weights between epsilon and 1
    
    # initial guess with equal weights and updated xi
    w0 = np.array(updated_w)
    
    for j in range(n):
        def obj(w, j=j):
            return w[j]
        
        result = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=constraints,     
            options={
                'ftol': 1e-9,       
                'disp': False,      
                'maxiter': 1000     
            }
        )

        max_weight = result[j]
        upper_weights.append(max_weight)

    return upper_weights

# ========================================
# Step 4: Calculate DP, P
# ========================================
def calculate_rank(n, low_weights, upper_weights):
    interval = list(zip(low_weights, upper_weights))

    def compare_preference(A, B):
        nominator = max(0, interval[A][1], interval[B][0]) - max(0, interval[A][0], interval[B][1])
        denominator = (interval[A][1] - interval[A][0]) + (interval[B][1] - interval[B][0])
        P_grt = nominator / denominator if denominator != 0 else 0

        if P_grt > 0.5:
            return P_grt
        elif P_grt < 0.5:
            return 1 - P_grt
        else:
            return 0.5    
          
    def calculate_similarity_matrix(n, interval):
        n = len(interval)
        DP = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    DP[i][j] = compare_preference(interval[i], interval[j])
        return DP
    
    def calculate_preference_matrix(n, DP):
        P = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if DP[i][j] > 0.5:
                    P[i][j] = 1
                else:
                    P[i][j] = 0
        return P
    
    def calculate_sum(n, preference_matrix):
        rank = []
        for i in range(n):
            rank.append(np.sum(preference_matrix[i]))
        return rank
    
    def statistics(rank):
        ci = np.std(rank) / np.mean(rank) if np.mean(rank) != 0 else 0
        ri = get_random_index(n)
        cr = ci / ri if ri != 0 else 0
        return ci, cr

    DP = calculate_similarity_matrix(n, interval)
    P = calculate_preference_matrix(n, DP)
    rank = calculate_sum(n, P)
    ci, cr = statistics(rank)
    return rank, ci, cr