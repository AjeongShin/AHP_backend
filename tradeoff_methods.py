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

    # calculate consistency
    wegiths_list = weights.tolist()
    inconsistency_ratios = []
    
    # for i in range(n):
    #     for j in range(n):
    #         inconsistency_ratios.append((wegiths_list[i] / wegiths_list[j]) / matrix[i][j])

    inconsistency_ratios = [
        [(wegiths_list[i] / wegiths_list[j]) / matrix[i][j] for j in range(n)]
        for i in range(n)
    ]
    return weights.tolist(), lambda_max, ci, cr, inconsistency_ratios

def triangular_fuzzy_ahp_solver(n, criteria, pairwise_matrix):
    """
    Parameters:
    -----------
    n : int
        Number of criteria
    criteria : list[str]
        List of criterion names
    pairwise_matrix : list[list[list[float]]]
        Pairwise comparison matrix as TFNs: [[[l11,c11,r11], [l12,c12,r12], ...], [...], ...]

    Returns:
    --------
    crisp_weights, lower_weights, upper_weights, score, sorted_criteria, 
    lambda_max, ci, cr, weights(TFN tuples), None    
    """

    # -----------------------------------------
    # Calculate coefficiet for Left, Center, Right
    # -----------------------------------------
    def calculate_coefficient(n, pairwise_matrix):
        coeff_min = np.zeros(n)
        coeff_max = np.zeros(n)
        
        for i in range(n):
            prod_l = 1.0
            prod_c = 1.0
            prod_u = 1.0

            for j in range(n):
                prod_l *= pairwise_matrix[i][j][0]  # l_ij
                prod_c *= pairwise_matrix[i][j][1]  # c_ij
                prod_u *= pairwise_matrix[i][j][2]  # r_ij
            coeff_min[i] = (prod_c ** (1/n)) / (prod_l ** (1/n))
            coeff_max[i] = (prod_c ** (1/n)) / (prod_u ** (1/n))
        
        c_min = np.min(coeff_min)
        c_max = np.max(coeff_max)
        
        return c_min, c_max

    # -----------------------------------------
    # Calculate weights from pairwise comparison matrix
    # -----------------------------------------
    def calculate_weights(n, pairwise_matrix, c_min, c_max):
        geomean_l = np.zeros(n)
        geomean_c = np.zeros(n)
        geomean_r = np.zeros(n)

        for i in range(n):
            prod_l = 1.0
            prod_c = 1.0
            prod_r = 1.0

            for j in range(n):
                prod_l *= pairwise_matrix[i][j][0]  # l_ij
                prod_c *= pairwise_matrix[i][j][1]  # c_ij
                prod_r *= pairwise_matrix[i][j][2]  # r_ßij

            geomean_l[i] = prod_l ** (1/n)
            geomean_c[i] = prod_c ** (1/n)
            geomean_r[i] = prod_r ** (1/n)

        sun_geomean_c = np.sum(geomean_c)

        weight_l = c_min * geomean_l / sun_geomean_c
        weight_c = geomean_c / sun_geomean_c
        weight_r = c_max * geomean_r / sun_geomean_c

        weights_tfn = list(zip(weight_l, weight_c, weight_r))
        return weight_l, weight_c, weight_r, weights_tfn

    
    # -----------------------------------------
    # Calculate ranking based on crisp weights
    # -----------------------------------------
    def calculate_rank(n, criteria, weight_c):
        crisp_weights = weight_c
        desc_indices = np.argsort(-crisp_weights)

        # find same rank criteria
        def is_same_rank(weights, desc_indices):
            final_indices = [] # [[idx1, idx2], [idx3], ...]
            # group = [] # [('critA','critB'), 'critC', ...]
            n = len(desc_indices)
            i = 0
            while i< n:
                current_idx = desc_indices[i]
                current_weights = weights[current_idx]
                group = [current_idx]
                j = i + 1
                while j < n and weights[desc_indices[j]] == current_weights:
                    group.append(desc_indices[j])
                    j += 1
                final_indices.append(group)
                i = j

            return final_indices
        

        final_indices = is_same_rank(crisp_weights, desc_indices)

        # transform to criteria 
        ordered_criteria = []
        for group in final_indices:
            if len(group) == 1:
                ordered_criteria.append(criteria[group[0]])
            else:
                ordered_criteria.append(tuple(criteria[idx] for idx in group))
        
        score = crisp_weights.copy()

        return score, ordered_criteria, crisp_weights
    
    # ========================================
    # Calculate lambda_max, CI(output-based), CR
    # ========================================
    def statistics(n, pairwise_matrix):

        A_center = np.array([[pairwise_matrix[i][j][1] for j in range(n)] for i in range(n)])
        _, lambda_max, ci, cr, _ = ahp_eigen_solver(A_center)

        return lambda_max, ci, cr
    
    #########################
    c_min, c_max = calculate_coefficient(n, pairwise_matrix)
    weight_l, weight_c, weight_r, weights_tfn = calculate_weights(n, pairwise_matrix, c_min, c_max)
    score, sorted_criteria, crisp_weights = calculate_rank(n, criteria, weight_c)
    lambda_max, ci, cr = statistics(n, pairwise_matrix)
    print("\n=== weights_tfn ===", weights_tfn)
    float_weights_tfn = [(float(l), float(c), float(r)) for l, c, r in weights_tfn]
    print("\n=== float_weights_tfn ===", float_weights_tfn)

    return crisp_weights, weight_l, weight_r, score, sorted_criteria, lambda_max, ci, cr, float_weights_tfn, None

def calculate_bwm_inconsistency(crisp_weights, best_idx, worst_idx, aB, aW, eps=1e-9):
    """
    AHP-style inconsistency ratios for BWM
    - use CENTER (crisp) weights
    - only declared comparisons (Best→Others, Others→Worst)
    """

    w = np.array(crisp_weights, dtype=float)

    wB = max(w[best_idx], eps)
    wW = max(w[worst_idx], eps)

    # Best → Others inconsistency
    bwo = []
    for j in range(len(w)):
        wj = max(w[j], eps)
        aBj = max(aB[j], eps)
        bwo.append((wB / wj) / aBj)

    # Others → Worst inconsistency
    wwo = []
    for i in range(len(w)):
        wi = max(w[i], eps)
        aWi = max(aW[i], eps)
        wwo.append((wi / wW) / aWi)

    return {
        "best_to_others": bwo,
        "others_to_worst": wwo
    }

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
    inconsistency_ratios = calculate_bwm_inconsistency(crisp_weights, best_idx, worst_idx, aB, aW)

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

    print("\n=== inconsistency_ratios ===")
    print(inconsistency_ratios)

    return crisp_weights, lower_weights, upper_weights, score, sorted_criteria, ci, cr, inconsistency_ratios 

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
    inconsistency_ratios = calculate_bwm_inconsistency(crisp_weights, best_idx, worst_idx, aB, aW)

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

    print("\n=== inconsistency_ratios ===")
    print(inconsistency_ratios)

    return crisp_weights, lower_weights, upper_weights, score, sorted_criteria, ci, cr, inconsistency_ratios

def triangular_fuzzy_bwm_solver(n, criteria, best_idx, worst_idx, aB, aW, epsilon=1e-6):
    """
    Parameters:
    -----------
    n : int
        Number of criteria
    criteria : list[str]
        List of criterion names
    best_idx : int
        Index of best criterion
    worst_idx : int
        Index of worst criterion
    aB : list[list[float]]
        Best-to-Others vector as TFNs: [[l1,m1,u1], [l2,m2,u2], ...]
    aW : list[list[float]]
        Others-to-Worst vector as TFNs: [[l1,m1,u1], [l2,m2,u2], ...]
    epsilon : float
        Small value to prevent division by zero
    
    Returns:
    --------
    crisp_weights, lower_weights, upper_weights, sorted_criteria, ci, cr
    """
    # -------------------------------
    # TFN Operations
    # -------------------------------
    def tfn_division(tfn1, tfn2, eps=1e-6):
        """TFN division: (l,m,u) / (l',m',u') = (l/u', m/m', u/l')"""
        return [tfn1[0] / max(tfn2[2], eps), tfn1[1] / max(tfn2[1], eps), tfn1[2] / max(tfn2[0], eps)]
    
    def tfn_subtraction(tfn1, tfn2):
        """TFN subtraction: (l1,m1,u1) - (l2,m2,u2) = (l1-u2, m1-m2, u1-l2)"""
        return [tfn1[0] - tfn2[2], tfn1[1] - tfn2[1], tfn1[2] - tfn2[0]]
    
    def tfn_graded_mean(tfn):
        """Graded mean: R(l,m,u) = (l + 4m + u) / 6"""
        return (tfn[0] + 4*tfn[1] + tfn[2]) / 6.0

    # -----------------------------------------
    # Step1: Find minimize k*, w* 
    # -----------------------------------------
    def minimize_k_fuzzybwm(n, best_idx, worst_idx, aB, aW, eps):
        # w: [w1,...,wn, k] = [l1,m1,u1, l2,m2,u2, ..., ln,mn,un, k*]
        # scipy.minimize requires flat 1D array

        def obj(w):  # minimize xi_R
            return w[-1]

        # Σ R(w_j) = 1
        def constraint_eq(w):
            total = 0.0
            for j in range(n):
                w_j = [w[3*j], w[3*j+1], w[3*j+2]]
                total += tfn_graded_mean(w_j)
            return total - 1.0

        constraints = [{'type': 'eq', 'fun': constraint_eq}]

        # l_j ≤ m_j ≤ u_j
        for j in range(n):
            def make_l_m_constraint(idx):
                def constraint(w):
                    return w[3*idx+1] - w[3*idx]
                return constraint
            
            def make_m_u_constraint(idx):
                def constraint(w):
                    return w[3*idx+2] - w[3*idx+1]
                return constraint
            
            constraints.append({'type': 'ineq', 'fun': make_l_m_constraint(j)})
            constraints.append({'type': 'ineq', 'fun': make_m_u_constraint(j)})
        

        for j in range(n):
            # Best-to-Others: |(w_B/w_j) - a_Bj| ≤ (k*,k*,k*)
            for s in range(3):  # s=0(l),1(m),2(u)
                def c_best_pos(w, j=j, s=s):
                    wB = [w[3*best_idx], w[3*best_idx+1], w[3*best_idx+2]]
                    wj = [w[3*j], w[3*j+1], w[3*j+2]]
                    k = w[-1]

                    division = tfn_division(wB, wj, eps)
                    return k - (division[s] - aB[j][s])
                
                def c_best_neg(w, j=j, s=s):
                    wB = [w[3*best_idx], w[3*best_idx+1], w[3*best_idx+2]]
                    wj = [w[3*j], w[3*j+1], w[3*j+2]]
                    k = w[-1]

                    division = tfn_division(wB, wj, eps)
                    return k + (division[s] - aB[j][s])

                # #################### vector form ####################
                # def c_best_pos(w, j=j):
                #     wB = np.array([w[3*best_idx], w[3*best_idx+1], w[3*best_idx+2]], dtype=float)
                #     wj = np.array([w[3*j+2], w[3*j+1], w[3*j]], dtype=float)
                #     k = w[-1]
                #     aBj = np.array(aB[j], dtype=float) 

                #     return k*wj - (wB - aBj*wj)


                # def c_best_neg(w, j=j):
                #     wB = np.array([w[3*best_idx], w[3*best_idx+1], w[3*best_idx+2]], dtype=float)
                #     wj = np.array([w[3*j+2], w[3*j+1], w[3*j]], dtype=float)
                #     k = w[-1]
                #     aBj = np.array(aB[j], dtype=float) 

                #     return k*wj + (wB - aBj*wj)
                

                constraints += [
                    {'type': 'ineq', 'fun': c_best_pos},
                    {'type': 'ineq', 'fun': c_best_neg},
                ]
                
            # Others-to-Worst: |(w_j/w_W) - a_jW| ≤ (k*,k*,k*)
            for s in range(3):  # s=0(l),1(m),2(u)
                def c_worst_pos(w, j=j, s=s):
                    wW = [w[3*worst_idx], w[3*worst_idx+1], w[3*worst_idx+2]]
                    wj = [w[3*j], w[3*j+1], w[3*j+2]]
                    k = w[-1]

                    division = tfn_division(wj, wW, eps)
                    return k - (division[s] - aW[j][s])

                def c_worst_neg(w, j=j, s=s):
                    wW = [w[3*worst_idx], w[3*worst_idx+1], w[3*worst_idx+2]]
                    wj = [w[3*j], w[3*j+1], w[3*j+2]]
                    k = w[-1]

                    division = tfn_division(wj, wW, eps)
                    return k + (division[s] - aW[j][s])

                # #################### vector form ####################
                # def c_worst_pos(w, j=j):
                #     wW = np.array([w[3*worst_idx+2], w[3*worst_idx+1], w[3*worst_idx]], dtype=float)
                #     wj = np.array([w[3*j], w[3*j+1], w[3*j+2]],dtype=float)
                #     k = w[-1]
                #     aWj = np.array(aW[j], dtype=float)

                #     return k*wW - (wj - aWj*wW)


                # def c_worst_neg(w, j=j):
                #     wW = np.array([w[3*worst_idx+2], w[3*worst_idx+1], w[3*worst_idx]], dtype=float)
                #     wj = np.array([w[3*j], w[3*j+1], w[3*j+2]],dtype=float)
                #     k = w[-1]
                #     aWj = np.array(aW[j], dtype=float)

                #     return k*wW + (wj - aWj*wW)

                
                constraints += [
                    {'type': 'ineq', 'fun': c_worst_pos},
                    {'type': 'ineq', 'fun': c_worst_neg},
                ]

        bounds = [(eps, 1.0) for _ in range(3*n)] + [(0.0, None)]  # w_i∈[eps,1], xi_R≥0
        w0 = np.array([1.0/n]*(3*n) + [1.0]) 

        res = minimize(
            obj, w0, method='SLSQP', bounds=bounds, constraints=constraints,
            options={'ftol': 1e-12, 'maxiter': 5000, 'disp': False}
        )
        # updated_w = res.x[:-1].tolist()
        updated_w = [[res.x[3*j], res.x[3*j+1], res.x[3*j+2]] for j in range(n)]
        k_star = float(res.x[-1])
        return updated_w, k_star

    # ========================================
    # Step 2: Calculate DP, P
    # ========================================
    def calculate_rank(n, criteria, updated_w):
        crisp_weights = np.array([tfn_graded_mean(updated_w[i]) for i in range(n)])
        desc_indices = np.argsort(-crisp_weights)

        # find same rank criteria
        def is_same_rank(weights, desc_indices):
            final_indices = [] # [[idx1, idx2], [idx3], ...]
            # group = [] # [('critA','critB'), 'critC', ...]
            n = len(desc_indices)
            i = 0
            while i< n:
                current_idx = desc_indices[i]
                current_weights = weights[current_idx]
                group = [current_idx]
                j = i + 1
                while j < n and weights[desc_indices[j]] == current_weights:
                    group.append(desc_indices[j])
                    j += 1
                final_indices.append(group)
                i = j

            return final_indices
        

        final_indices = is_same_rank(crisp_weights, desc_indices)

        # transform to criteria 
        ordered_criteria = []
        for group in final_indices:
            if len(group) == 1:
                ordered_criteria.append(criteria[group[0]])
            else:
                ordered_criteria.append(tuple(criteria[idx] for idx in group))
        
        score = crisp_weights.copy()

        return score, ordered_criteria, crisp_weights

    # ========================================
    # Step 2: Calculate crisp weight, CI(output-based), CR
    # ========================================
    def statistics(n, k_star, aB, worst_idx, updated_w, crisp_weights):

        def get_consistency_index_fuzzy_bwm(aB, worst_idx):
            SCALE = {
                'EI': (1.0, 1.0, 1.0),
                'WI': (2/3, 1.0, 3/2),
                'FI': (3/2, 2.0, 5/2),
                'VI': (5/2, 3.0, 7/2),
                'AI': (7/2, 4.0, 9/2),
            }

            """BWM Consistency Index lookup table"""
            ci_table = {
                'EI': 3.00, 'WI': 3.80, 'FI': 5.29, 'VI': 6.69, 'AI': 8.04
            }

            a_BW_value = aB[worst_idx]
            a_BW_label = None

            for label, tfn in SCALE.items():
                if np.allclose(a_BW_value, tfn, atol=1e-6):
                    a_BW_label = label
                    break

            return ci_table.get(a_BW_label, None), a_BW_label

        ci, _ = get_consistency_index_fuzzy_bwm(aB, worst_idx)
        if ci != None:
            cr = k_star / ci if ci != 0 else 0
        else:
            cr = None # if ci is None, cr is also None
        return ci, cr, crisp_weights
    
    #########################
    # solve optimization prob(linear BWM) by using scipy.linprog 
    # return value would be same with AHP(wiehg, ci, cr ..)
    #######################
    updated_w, k_star = minimize_k_fuzzybwm(n, best_idx, worst_idx, aB, aW, epsilon)
    score, sorted_criteria, crisp_weights = calculate_rank(n, criteria, updated_w)
    ci, cr, crisp_weights = statistics(n, k_star, aB, worst_idx, updated_w, crisp_weights)

    # debugging print
    print("\n=== updated_w ===")
    print(updated_w)

    print("\n=== k_star ===")
    print(k_star)

    print("\n=== ci ===")
    print(ci)

    print("\n=== cr ===")
    print(cr)

    print("\n=== crisp_weights ===")
    print(crisp_weights)

    return crisp_weights, None, None, score, sorted_criteria, ci, cr, updated_w, k_star

        