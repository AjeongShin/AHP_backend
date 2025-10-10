from flask import Flask, request, jsonify
from flask_cors import CORS
from tradeoff_methods import ahp_eigen_solver
from tradeoff_methods import linear_bwm_solver

import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/ahp_calculate', methods=['POST'])
def ahp_calculate():
    data = request.get_json()
    if not data or 'matrix' not in data:
        return jsonify({'error': 'No matrix provided'}), 400

    try:
        # Remove rows containing 0 (invalid criteria)
        raw_matrix = np.array(data['matrix'])
        valid_indices = [i for i in range(len(raw_matrix)) if all(val != 0 for val in raw_matrix[i])]
        matrix = raw_matrix[np.ix_(valid_indices, valid_indices)]

        # Use Eigenvector method for calculation
        weights, lambda_max, ci, cr = ahp_eigen_solver(matrix)

        return jsonify({
            'weights': weights,
            'lambdaMax': lambda_max,
            'ci': ci,
            'cr': cr
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/bwm_calculate', methods=['POST'])
def bwm_calculate():
    data = request.get_json()
    # if not data or 'matrix' not in data:
    #     return jsonify({'error': 'No matrix provided'}), 400

    try:
        # Parcing dictionary to element
        n = int(data['n'])
        criteria = data['criteria']
        best_idx = int(data['bestIdx'])
        worst_idx = int(data['worstIdx'])
        aB = np.array(data['bestRow'], dtype=float)
        aW = np.array(data['worstCol'], dtype=float)

        # Use Eigenvector method for calculation
        weights, score, sorted_criteria, ci, cr = linear_bwm_solver(n, criteria, best_idx, worst_idx, aB, aW, epsilon=1e-6)
        
        payload = {
            'weights': weights,
            'score': score,
            'sorted_criteria': sorted_criteria,
            'ci': ci,
            'cr': cr,
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