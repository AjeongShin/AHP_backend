from flask import Flask, request, jsonify
from flask_cors import CORS
from tradeoff_methods import ahp_eigen_solver, triangular_fuzzy_ahp_solver
from tradeoff_methods import linear_bwm_solver, non_linear_bwm_solver, triangular_fuzzy_bwm_solver

import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/ahp_calculate', methods=['POST'])
def ahp_calculate():
    data = request.get_json()
    if not data or 'matrix' not in data:
        return jsonify({'error': 'No matrix provided'}), 400

    try:
        # # Remove rows containing 0 (invalid criteria)
        # raw_matrix = np.array(data['matrix'])
        # valid_indices = [i for i in range(len(raw_matrix)) if all(val != 0 for val in raw_matrix[i])]
        # matrix = raw_matrix[np.ix_(valid_indices, valid_indices)]

        # Parcing dictionary to element
        matrix = np.array(data['matrix'])
        ahp_variant = data['variant']
        n = int(data['n'])
        criteria = data['criteria']
        # best_idx = int(data['bestIdx'])
        # worst_idx = int(data['worstIdx'])

        lower_weights, upper_weights = None, None
        updated_w, k_star = None, None
        inconsistency_ratios = None

        # Use Eigenvector method for calculation
        if ahp_variant == 'origin':
            crisp_weights, lambda_max, ci, cr, inconsistency_ratios = ahp_eigen_solver(matrix)
            score, sorted_criteria = None, None
        else: # 'fuzzy' (triangular_fuzzy_ahp_solver)
            # print("before solver", flush=True)
            # print("matrix:", matrix, type(matrix), flush=True)
            crisp_weights, lower_weights, upper_weights, score, sorted_criteria, lambda_max, ci, cr, updated_w, k_star = triangular_fuzzy_ahp_solver(n, criteria, matrix)
            # print("updated_w:", updated_w, type(updated_w))
            if lower_weights is None and upper_weights is None and isinstance(updated_w, list):
                lower_weights = [float(t[0]) for t in updated_w]
                upper_weights = [float(t[2]) for t in updated_w]

        if isinstance(crisp_weights, np.ndarray):
            crisp_weights = crisp_weights.tolist()
        if isinstance(lower_weights, np.ndarray):
            lower_weights = lower_weights.tolist()
        if isinstance(upper_weights, np.ndarray):
            upper_weights = upper_weights.tolist()
        if isinstance(score, np.ndarray):
            score = score.tolist()
        if isinstance(inconsistency_ratios, np.ndarray):
            inconsistency_ratios = inconsistency_ratios.tolist()


        payload = {
            'crisp_weights': crisp_weights,
            'lower_weights': lower_weights,
            'upper_weights': upper_weights,
            'score': score,
            'sorted_criteria': sorted_criteria,
            'lambdaMax': lambda_max,
            'ci': ci,
            'cr': cr,
            'inconsistency_ratios': inconsistency_ratios,
            'extra':{
                'tfn_weights': updated_w,
                'k_star': k_star,
            }
        }
        return jsonify(payload)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/bwm_calculate', methods=['POST'])
def bwm_calculate():
    data = request.get_json()
    # if not data or 'matrix' not in data:
    #     return jsonify({'error': 'No matrix provided'}), 400

    try:
        # Parcing dictionary to element
        bwm_variant = data['variant']
        n = int(data['n'])
        criteria = data['criteria']
        best_idx = int(data['bestIdx'])
        worst_idx = int(data['worstIdx'])
        aB = np.array(data['bestRow'], dtype=float)
        aW = np.array(data['worstCol'], dtype=float)
        # extra = {}

        lower_weights, upper_weights = None, None
        updated_w, k_star = None, None

        # Use Eigenvector method for calculation
        if bwm_variant == 'linear':
            crisp_weights, lower_weights, upper_weights, score, sorted_criteria, ci, cr = linear_bwm_solver(n, criteria, best_idx, worst_idx, aB, aW, epsilon=1e-6)
        elif bwm_variant == 'nonlinear':
            crisp_weights, lower_weights, upper_weights, score, sorted_criteria, ci, cr = non_linear_bwm_solver(n, criteria, best_idx, worst_idx, aB, aW, epsilon=1e-6)
        # else:
        #     crisp_weights, _, _, score, sorted_criteria, ci, cr, updated_w, k_star = triangular_fuzzy_bwm_solver(n, criteria, best_idx, worst_idx, aB, aW, epsilon=1e-6)
        else: # 'fuzzy' (triangular_fuzzy_bwm_solver)
            crisp_weights, lower_weights, upper_weights, score, sorted_criteria, ci, cr, updated_w, k_star = triangular_fuzzy_bwm_solver(n, criteria, best_idx, worst_idx, aB, aW, epsilon=1e-6)

            if lower_weights is None and upper_weights is None and isinstance(updated_w, list):
                lower_weights = [float(t[0]) for t in updated_w]
                upper_weights = [float(t[2]) for t in updated_w]

        if isinstance(crisp_weights, np.ndarray):
            crisp_weights = crisp_weights.tolist()
        if isinstance(score, np.ndarray):
            score = score.tolist()

        payload = {
            'crisp_weights': crisp_weights,
            'lower_weights': lower_weights,
            'upper_weights': upper_weights,
            'score': score,
            'sorted_criteria': sorted_criteria,
            'ci': ci,
            'cr': cr,
            'extra':{
                'tfn_weights': updated_w,
                'k_star': k_star,
            }
        }

        return jsonify(payload)
    except Exception as e:
        # return jsonify({'error': str(e)}), 500
        return jsonify({'error': repr(e)}), 500

# Confirm the server is running    
@app.route('/', methods=['GET'])
def home():
    return 'AHP API is running.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)