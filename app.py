from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Function to get RI (Random Index) value
def get_random_index(n):
    ri_table = {
        1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
    }
    return ri_table.get(n, 1.49)

# Calculation function based on the Eigenvector method
def calculate_ahp_eigen(matrix):
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

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    if not data or 'matrix' not in data:
        return jsonify({'error': 'No matrix provided'}), 400

    try:
        # Remove rows containing 0 (invalid criteria)
        raw_matrix = np.array(data['matrix'])
        valid_indices = [i for i in range(len(raw_matrix)) if all(val != 0 for val in raw_matrix[i])]
        matrix = raw_matrix[np.ix_(valid_indices, valid_indices)]

        # Use Eigenvector method for calculation
        weights, lambda_max, ci, cr = calculate_ahp_eigen(matrix)

        return jsonify({
            'weights': weights,
            'lambdaMax': lambda_max,
            'ci': ci,
            'cr': cr
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # local host
    # app.run(host='127.0.0.1', port=5000, debug=True)
    app.run(host='0.0.0.0', port=8080)