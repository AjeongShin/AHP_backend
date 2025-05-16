# AHP Pairwise Calculator

A lightweight API that calculates **priority weights** and evaluates **consistency** from an Analytic Hierarchy Process (AHP) pairwise comparison matrix using the **Eigenvector Method**.

---

## Features

- Accepts AHP pairwise matrix via JSON input
- Removes criteria with invalid (zero) values
- Computes:
  - Normalized weights from the principal eigenvector
  - λₘₐₓ (principal eigenvalue)
  - CI (Consistency Index)
  - CR (Consistency Ratio)
- Stateless and does not use a database
- Easily deployable to Google Cloud or other platforms

---

## Formulas

Let **A** be an n×n pairwise comparison matrix.

### Eigen Decomposition

Solve:  
```
A * v = λ * v
```
Select eigenvector **v<sub>max</sub>** corresponding to the largest real eigenvalue **λ<sub>max</sub>**

### Normalize Eigenvector
```
wᵢ = |vᵢ| / Σ|vⱼ|
```
Where **w** is the final weight vector.

### Consistency Index (CI)
```
CI = (λₘₐₓ - n) / (n - 1)
```

### Consistency Ratio (CR)
```
CR = CI / RI
```

Where RI (Random Index) is:

| n  | RI   |
|----|------|
| 1  | 0.00 |
| 2  | 0.00 |
| 3  | 0.58 |
| 4  | 0.90 |
| 5  | 1.12 |
| 6  | 1.24 |
| 7  | 1.32 |
| 8  | 1.41 |
| 9  | 1.45 |
| 10 | 1.49 |

---

## Local Deployment

### 1. Install dependencies

```bash
pip install flask flask-cors numpy
```

### 2. Run the Flask server

```bash
python app.py
```

You should see output like:

```
 * Running on http://127.0.0.1:8080
```

---

## ✅ Verify It Works

### Option 1: Test using `curl`

Open a new terminal and run:

```bash
curl -X POST http://127.0.0.1:8080/calculate \
-H "Content-Type: application/json" \
-d '{"matrix": [[1,3,0.5],[0.33,1,0.25],[2,4,1]]}'
```

Expected output:

```json
{
  "ci": 0.007690179197919589,
  "cr": 0.013258929651585498,
  "lambdaMax": 3.015380358395839,
  "weights": [0.3196792344065948, 0.12164594688534666, 0.5586748187080586]
}
```

### Option 2: Add a root route (for health check)

Add this to `app.py`:

```python
@app.route('/', methods=['GET'])
def home():
    return 'AHP API is running.'
```

Then visit [http://127.0.0.1:8080](http://127.0.0.1:8080) in your browser.

---

## Deployment Notes

This app is deployed using **Google Cloud Build + Cloud Run**.

> [Live API Endpoint](https://ahp-backend-725147247515.europe-west1.run.app)

You can `POST` to `/calculate` on that domain just like the local example.

## License

MIT License
---
