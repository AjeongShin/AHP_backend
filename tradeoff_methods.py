import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

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

    def create_constraints(n, best_idx, worst_idx, aB, aW, fixed_xi=None, use_xi_variable=True, epsilon=1e-6):
        """
        Parameters:
        -----------
        fixed_xi : float, optional
            use fixed xi_L value (Step2, Step3)
        use_xi_variable : bool
            w vector includes xi_L value (Step1: True, Step2/3: False)
        """
        # equality constraint: sum of weights = 1 (f(x) = 0)
        def constraint_eq(w):
            if use_xi_variable:
                return np.sum(w[:-1]) - 1  # w = [w1,...,wn, xi]
            else:
                return np.sum(w) - 1  # w = [w1,...,wn]
        
        constraints = [{'type':'eq', 'fun': constraint_eq}]

        for j in range(n):
            # Linear BWM Constraints:
            # |w_B - a_Bj*w_j| ≤ ξ_L  →  two inequalities
            # |w_j - a_jW*w_W| ≤ ξ_L  →  two inequalities

            def constraint_ineq_best_positive(j=j):
                def constraint(w):
                    if use_xi_variable:
                        weights = w[:-1]
                        xi_L = w[-1]
                    else:
                        weights = w
                        xi_L = fixed_xi
                    weight_best = weights[best_idx]
                    weight_j = weights[j]
                    # w_B - a_Bj*w_j ≤ ξ_L
                    # → ξ_L - (w_B - a_Bj*w_j) ≥ 0

                    # linear version
                    return xi_L - (weight_best - aB[j]*weight_j)
                return constraint
                
            def constraint_ineq_best_negative(j=j):
                def constraint(w):
                    if use_xi_variable:
                        weights = w[:-1]
                        xi_L = w[-1]
                    else:
                        weights = w
                        xi_L = fixed_xi
                    weight_best = weights[best_idx]
                    weight_j = weights[j]
                    # -(w_B - a_Bj*w_j) ≤ ξ_L
                    # → a_Bj*w_j - w_B ≤ ξ_L
                    # → ξ_L - (a_Bj*w_j - w_B) ≥ 0

                    # linear version
                    return xi_L + (weight_best - aB[j]*weight_j)
                return constraint

            
            def constraint_ineq_worst_positive(j=j):
                def constraint(w):
                    if use_xi_variable:
                        weights = w[:-1]
                        xi_L = w[-1]
                    else:
                        weights = w
                        xi_L = fixed_xi
                    weight_worst = weights[worst_idx]
                    weight_j = weights[j]
                    # w_j - a_jW*w_W ≤ ξ_L
                    # → ξ_L - (w_j - a_jW*w_W) ≥ 0

                    # linear version
                    return xi_L - (weight_j - aW[j]*weight_worst)
                return constraint

            
            def constraint_ineq_worst_negative(j=j):
                def constraint(w):
                    if use_xi_variable:
                        weights = w[:-1]
                        xi_L = w[-1]
                    else:
                        weights = w
                        xi_L = fixed_xi
                    weight_worst = weights[worst_idx]
                    weight_j = weights[j]
                    # -(w_j - a_jW*w_W) ≤ ξ_L
                    # → a_jW*w_W - w_j ≤ ξ_L
                    # → ξ_L - (a_jW*w_W - w_j) ≥ 0

                    # linear version
                    return xi_L + (weight_j - aW[j]*weight_worst)
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
        constraints = create_constraints(n, best_idx, worst_idx, aB, aW,
            fixed_xi=None, use_xi_variable=True, epsilon=epsilon )
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
        updated_w = result.x[:n].tolist()
        updated_xi = result.x[-1]

        return updated_w, updated_xi

    # ========================================
    # Step 2: Lower Bound Weights
    # ========================================
    def lower_bound_weights(n, best_idx, worst_idx, aB, aW, updated_w, updated_xi, epsilon=1e-4):
        lower_weights = []

        # constraints of equality and inequality
        constraints = create_constraints(n, best_idx, worst_idx, aB, aW, fixed_xi=updated_xi, use_xi_variable=False, epsilon=epsilon )

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
                    'maxiter': 1000                    }
            )

            min_weight = result.x[j]
            lower_weights.append(min_weight)

        return lower_weights

    # ========================================
    # Step 3: Upper Bound Weights
    # ========================================
    def maximize_weights(n, best_idx, worst_idx, aB, aW, updated_w, updated_xi, epsilon=1e-4):
        upper_weights = []

        # constraints of equality and inequality
        constraints = create_constraints(n, best_idx, worst_idx, aB, aW, fixed_xi=updated_xi, use_xi_variable=False, epsilon=epsilon )

        # bounds for weights and xi
        # bounds = [(epsilon, None) for _ in range(n)]  # weights between epsilon and 1
        bounds = [(0, 1) for _ in range(n)]
        
        # initial guess with equal weights and updated xi
        w0 = np.array(updated_w)
        
        for j in range(n):
            def obj(w, j=j):
                return -w[j]
            
            result = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=constraints,     
                options={
                    'ftol': 1e-9,       
                    'disp': False,      
                    'maxiter': 1000     
                }
            )

            max_weight = result.x[j]
            upper_weights.append(max_weight)

        return upper_weights

    # ========================================
    # Step 4: Calculate DP, P
    # ========================================
    def calculate_rank(n, criteria, low_weights, upper_weights):
        interval = list(zip(low_weights, upper_weights))

        def compare_preference(interval_A, interval_B):
            a_L, a_R = interval_A
            b_L, b_R = interval_B

            numerator = max(0, a_R - b_L) - max(0, a_L - b_R)
            denominator = (a_R - a_L) + (b_R - b_L)
            P_a_grt = numerator / denominator if denominator != 0 else 0

            return P_a_grt    
            
        def calculate_similarity_matrix(n, interval):
            n = len(interval)
            DP = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
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
        
        def calculate_sum(criteria, preference_matrix, upper_weights, tol=1e-9):
            score = np.sum(preference_matrix, axis=1)
            if np.allclose(score, 0.0, atol=tol):
                score = np.asarray(upper_weights, dtype=float) # if all score is 0, use upper weights for ranking
            desc_indices = np.argsort(-score)

            # find same rank criteria
            def is_same_rank(score, desc_indices):
                final_indices = [] # [[idx1, idx2], [idx3], ...]
                # group = [] # [('critA','critB'), 'critC', ...]
                n = len(desc_indices)
                i = 0
                while i< n:
                    current_idx = desc_indices[i]
                    current_score = score[current_idx]
                    group = [current_idx]
                    j = i + 1
                    while j < n and score[desc_indices[j]] == current_score:
                        group.append(desc_indices[j])
                        j += 1
                    final_indices.append(group)
                    i = j

                return final_indices
            

            final_indices = is_same_rank(score, desc_indices)

            # transform to criteria 
            ordered_criteria = []
            for group in final_indices:
                if len(group) == 1:
                    ordered_criteria.append(criteria[group[0]])
                else:
                    ordered_criteria.append(tuple(criteria[idx] for idx in group))
            
            return score, ordered_criteria
        
        DP = calculate_similarity_matrix(n, interval)
        P = calculate_preference_matrix(n, DP)
        score, order_criteria = calculate_sum(criteria, P, upper_weights, tol=1e-9)
        if isinstance(score, np.ndarray):
            score = score.tolist() # .tolist() for jsonify

        return DP, P, score, order_criteria

    # ========================================
    # Step 5: Calculate CI(output-based), CR
    # ========================================
    def statistics(updated_xi, aB, worst_idx, lower_weights, upper_weights):

        def get_consistency_index_bwm(aB, worst_idx):
            a_BW = aB[worst_idx]

            """BWM Consistency Index lookup table"""
            ci_table = {
                1: 0.00, 2: 0.44, 3: 1.00, 4: 1.63, 5: 2.30,
                6: 3.00, 7: 3.73, 8: 4.47, 9: 5.23
            }
            return ci_table.get(a_BW, None), a_BW # if mode_result > 9, return None
        """
        BWM Consistency Ratio calculation
        
        Parameters:
        -----------
        updated_xi : float
            Optimal xi value (ξ*)
        a_BW : int
            Best-to-Worst comparison value
        
        Returns:
        --------
        ci : float
            Consistency Index from lookup table
        cr : float
            Consistency Ratio = ξ* / CI
        """

        center_weights = [0.5*(lo+up) for lo, up in zip(lower_weights, upper_weights)]

        ci, a_BW = get_consistency_index_bwm(aB, worst_idx)
        if ci != None:
            cr = updated_xi / ci if ci != 0 else 0
        else:
            cr = None # if ci is None, cr is also None
        return ci, cr, a_BW, updated_xi, center_weights
    
    #########################
    # solve optimization prob(linear BWM) by using scipy.linprog 
    # return value would be same with AHP(wiehg, ci, cr ..)
    ########################
    updated_w, updated_xi = minimize_xi(n, best_idx, worst_idx, aB, aW, epsilon)
    lower_weights = lower_bound_weights(n, best_idx, worst_idx, aB, aW, updated_w, updated_xi, epsilon)
    upper_weights = maximize_weights(n, best_idx, worst_idx, aB, aW, updated_w, updated_xi, epsilon)
    DP, P, score, sorted_criteria = calculate_rank(n, criteria, lower_weights, upper_weights)
    ci, cr, a_BW, updated_xi, crisp_weights = statistics(updated_xi, aB, worst_idx, lower_weights, upper_weights)

    # debugging print
    print("\n=== a_BW ===")
    print(a_BW)

    print("\n=== updated_xi ===")
    print(updated_xi)

    print("\n=== lower_weights ===")
    print(lower_weights)

    print("\n=== upper_weights ===")
    print(upper_weights)

    print("\n=== crisp_weights ===")
    print(crisp_weights)

    return crisp_weights, lower_weights, upper_weights, score, sorted_criteria, ci, cr

def non_linear_bwm_solver(n, criteria, best_idx, worst_idx, aB, aW, epsilon=1e-6):
    # -------------------------------
    # Solve Nonlinear BWM optimization problem.
    # |w_B/w_j - a_Bj| ≤ ξ, |w_j/w_W - a_jW| ≤ ξ, sum(w)=1
    # -------------------------------
    def create_constraints_bilinear(n, best_idx, worst_idx, aB, aW, fixed_xi, eps):
        def constraint_eq(w):
            return np.sum(w) - 1.0

        constraints = [{'type': 'eq', 'fun': constraint_eq}]

        for j in range(n):
            def c_best_pos(w, j=j):
                wB = w[best_idx]
                wj = max(w[j], eps)
                return fixed_xi - (wB / wj - aB[j])

            def c_best_neg(w, j=j):
                wB = w[best_idx]
                wj = max(w[j], eps)
                return fixed_xi + (wB / wj - aB[j])

            def c_worst_pos(w, j=j):
                wW = max(w[worst_idx], eps)
                wj = w[j]
                return fixed_xi - (wj / wW - aW[j])

            def c_worst_neg(w, j=j):
                wW = max(w[worst_idx], eps)
                wj = w[j]
                return fixed_xi + (wj / wW - aW[j])

            constraints += [
                {'type': 'ineq', 'fun': c_best_pos},
                {'type': 'ineq', 'fun': c_best_neg},
                {'type': 'ineq', 'fun': c_worst_pos},
                {'type': 'ineq', 'fun': c_worst_neg},
            ]
        return constraints

    # -----------------------------------------
    # Minimize ξ_R*, w*
    # -----------------------------------------
    def minimize_xi_bilinear(n, best_idx, worst_idx, aB, aW, eps):
        # [w1,...,wn, xi_R]
        def obj(w_all):  # minimize xi_R
            return w_all[-1]

        def constraint_eq(w_all):
            return np.sum(w_all[:-1]) - 1.0

        constraints = [{'type': 'eq', 'fun': constraint_eq}]

        for j in range(n):
            def c_best_pos(w_all, j=j):
                wB = w_all[best_idx]
                wj = max(w_all[j], eps)
                xi = w_all[-1]
                return xi - (wB / wj - aB[j])

            def c_best_neg(w_all, j=j):
                wB = w_all[best_idx]
                wj = max(w_all[j], eps)
                xi = w_all[-1]
                return xi + (wB / wj - aB[j])

            def c_worst_pos(w_all, j=j):
                wW = max(w_all[worst_idx], eps)
                wj = w_all[j]
                xi = w_all[-1]
                return xi - (wj / wW - aW[j])

            def c_worst_neg(w_all, j=j):
                wW = max(w_all[worst_idx], eps)
                wj = w_all[j]
                xi = w_all[-1]
                return xi + (wj / wW - aW[j])

            constraints += [
                {'type': 'ineq', 'fun': c_best_pos},
                {'type': 'ineq', 'fun': c_best_neg},
                {'type': 'ineq', 'fun': c_worst_pos},
                {'type': 'ineq', 'fun': c_worst_neg},
            ]

        bounds = [(eps, 1.0) for _ in range(n)] + [(0.0, None)]  # w_i∈[eps,1], xi_R≥0
        w0 = np.array([1.0/n]*n + [1.0]) 

        res = minimize(
            obj, w0, method='SLSQP', bounds=bounds, constraints=constraints,
            options={'ftol': 1e-12, 'maxiter': 5000, 'disp': False}
        )
        updated_w = res.x[:-1].tolist()
        xiR_star = float(res.x[-1])
        return updated_w, xiR_star

    # -----------------------------------------
    # Perform multi-start optimization to find global optimum.
    # -----------------------------------------
    def multi_start_best_x(obj, w0_list, bounds, constraints):
        best_x, best_f = None, None
        for w0 in w0_list:
            res = minimize(
                obj, w0, method='SLSQP', bounds=bounds, constraints=constraints,
                options={'ftol': 1e-12, 'maxiter': 5000, 'disp': False}
            )
            if best_f is None or res.fun < best_f:
                best_f, best_x = res.fun, res.x
        return best_x

    # ========================================
    # Step 1: Minimize Xi
    # ========================================
    updated_w, updated_xi = minimize_xi_bilinear(n, best_idx, worst_idx, aB, aW, epsilon)

    # ========================================
    # Step 2: Lower Bound 
    # ========================================
    def lower_bound_weights(n, best_idx, worst_idx, aB, aW, updated_w, updated_xi, eps=epsilon):
        lower_weights = []
        constraints = create_constraints_bilinear(n, best_idx, worst_idx, aB, aW, fixed_xi=updated_xi, eps=eps)
        bounds = [(eps, 1.0) for _ in range(n)]

        for j in range(n):
            # Strategy 1: Start from optimal weights
            w0a = np.array(updated_w, dtype=float)

            # Strategy 2: Make w_j small, redistribute rest proportionally
            w0b = np.array(updated_w, dtype=float)
            target = max(eps, updated_w[j]*0.2)
            rest = 1.0 - target
            others = [k for k in range(n) if k != j]
            scale = rest / np.sum(w0b[others])
            w0b[others] = w0b[others] * scale
            w0b[j] = target

            best_x = multi_start_best_x(lambda w, j=j: w[j], [w0a, w0b], bounds, constraints)
            lower_weights.append(float(best_x[j]))
        return lower_weights

    # ========================================
    # [CHANGED] Step 3: Upper Bound 
    # ========================================
    def maximize_weights(n, best_idx, worst_idx, aB, aW, updated_w, updated_xi, eps=epsilon):
        upper_weights = []
        constraints = create_constraints_bilinear(n, best_idx, worst_idx, aB, aW, fixed_xi=updated_xi, eps=eps)
        bounds = [(eps, 1.0) for _ in range(n)]

        for j in range(n):
            # Strategy 1: Start from optimal weights
            w0a = np.array(updated_w, dtype=float)

            # Strategy 2: Make w_j large, shrink rest proportionally
            w0b = np.array(updated_w, dtype=float)
            target = min(0.9, updated_w[j]*1.8)
            rest = 1.0 - target
            others = [k for k in range(n) if k != j]
            scale = rest / np.sum(w0b[others])
            w0b[others] = w0b[others] * scale
            w0b[j] = target

            best_x = multi_start_best_x(lambda w, j=j: -w[j], [w0a, w0b], bounds, constraints)
            upper_weights.append(float(best_x[j]))
        return upper_weights

    # ========================================
    # Step 4: Calculate DP, P
    # ========================================
    def calculate_rank(n, criteria, low_weights, upper_weights):
        interval = list(zip(low_weights, upper_weights))

        def compare_preference(interval_A, interval_B):
            a_L, a_R = interval_A
            b_L, b_R = interval_B

            numerator = max(0, a_R - b_L) - max(0, a_L - b_R)
            denominator = (a_R - a_L) + (b_R - b_L)
            P_a_grt = numerator / denominator if denominator != 0 else 0

            return P_a_grt    
            
        def calculate_similarity_matrix(n, interval):
            n = len(interval)
            DP = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
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
        
        def calculate_sum(criteria, preference_matrix, upper_weights, tol=1e-9):
            score = np.sum(preference_matrix, axis=1)
            if np.allclose(score, 0.0, atol=tol):
                score = np.asarray(upper_weights, dtype=float) # if all score is 0, use upper weights for ranking
            desc_indices = np.argsort(-score)

            # find same rank criteria
            def is_same_rank(score, desc_indices):
                final_indices = [] # [[idx1, idx2], [idx3], ...]
                # group = [] # [('critA','critB'), 'critC', ...]
                n = len(desc_indices)
                i = 0
                while i< n:
                    current_idx = desc_indices[i]
                    current_score = score[current_idx]
                    group = [current_idx]
                    j = i + 1
                    while j < n and score[desc_indices[j]] == current_score:
                        group.append(desc_indices[j])
                        j += 1
                    final_indices.append(group)
                    i = j

                return final_indices
            

            final_indices = is_same_rank(score, desc_indices)

            # transform to criteria 
            ordered_criteria = []
            for group in final_indices:
                if len(group) == 1:
                    ordered_criteria.append(criteria[group[0]])
                else:
                    ordered_criteria.append(tuple(criteria[idx] for idx in group))
            
            return score, ordered_criteria
        
        DP = calculate_similarity_matrix(n, interval)
        P = calculate_preference_matrix(n, DP)
        score, order_criteria = calculate_sum(criteria, P, upper_weights, tol=1e-9)
        if isinstance(score, np.ndarray):
            score = score.tolist() # .tolist() for jsonify

        return DP, P, score, order_criteria

    # ========================================
    # Step 5: Calculate CI(output-based), CR
    # ========================================
    def statistics(updated_xi, aB, worst_idx, lower_weights, upper_weights):

        def get_consistency_index_bwm(aB, worst_idx):
            a_BW = aB[worst_idx]

            """BWM Consistency Index lookup table"""
            ci_table = {
                1: 0.00, 2: 0.44, 3: 1.00, 4: 1.63, 5: 2.30,
                6: 3.00, 7: 3.73, 8: 4.47, 9: 5.23
            }
            return ci_table.get(a_BW, None), a_BW # if mode_result > 9, return None
        """
        BWM Consistency Ratio calculation
        
        Parameters:
        -----------
        updated_xi : float
            Optimal xi value (ξ*)
        a_BW : int
            Best-to-Worst comparison value
        
        Returns:
        --------
        ci : float
            Consistency Index from lookup table
        cr : float
            Consistency Ratio = ξ* / CI
        """

        center_weights = [0.5*(lo+up) for lo, up in zip(lower_weights, upper_weights)]

        ci, a_BW = get_consistency_index_bwm(aB, worst_idx)
        if ci != None:
            cr = updated_xi / ci if ci != 0 else 0
        else:
            cr = None # if ci is None, cr is also None
        return ci, cr, a_BW, updated_xi, center_weights
    
    #########################
    # solve optimization prob(linear BWM) by using scipy.linprog 
    # return value would be same with AHP(wiehg, ci, cr ..)
    ########################
    lower_weights = lower_bound_weights(n, best_idx, worst_idx, aB, aW, updated_w, updated_xi)
    upper_weights = maximize_weights(n, best_idx, worst_idx, aB, aW, updated_w, updated_xi)
    DP, P, score, sorted_criteria = calculate_rank(n, criteria, lower_weights, upper_weights)
    ci, cr, a_BW, updated_xi, crisp_weights = statistics(updated_xi, aB, worst_idx, lower_weights, upper_weights)

    # debugging print
    # print("\n=== DP Matrix ===")
    # df_DP = pd.DataFrame(DP, index=criteria, columns=criteria)
    # print(df_DP)
    # print("="*50 + "\n")

    # print("\n=== Preference Matrix (P) ===")
    # df_P = pd.DataFrame(P, index=criteria, columns=criteria)
    # print(df_P)
    # print("="*50 + "\n")

    print("\n=== a_BW ===")
    print(a_BW)

    print("\n=== updated_xi ===")
    print(updated_xi)

    print("\n=== lower_weights ===")
    print(lower_weights)

    print("\n=== upper_weights ===")
    print(upper_weights)

    print("\n=== crisp_weights ===")
    print(crisp_weights)



    return crisp_weights, lower_weights, upper_weights, score, sorted_criteria, ci, cr
