# Human Activity Recognition Exercises — English version

**.objective**: Design a set of 6 lightweight practical exercises (about 1-3 hours each) on Human Activity Recognition (HAR), using data that students can easily access. The project should be self-testable by later students, and optionally allow students to use their own phone data or even create a mobile app component.

---

## Dataset Options

| Source | Description | Size / Effort | Link |
|---|---|---|---|
| **UCI minimal accelerometer** | 2 activities (standing/walking), 6 features (accX,accY,accZ,gyroX,gyroY,gyroZ), 100 rows × 6 cols | 2 MB, immediate `ucimlrepo` or manual CSV | https://archive.ics.uci.edu/dataset/755/accelerometer+gyro+mobile+phone+dataset |
| **UCI full HAR** | 30 subjects, 6 activities, 561 features (from smartphone sensors) | ~30 MB, more rich but more columns | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+on+smartphones |
| **MDPI article dataset** (HARTH) | Free-living accelerometer dataset, 22 participants, annotated daily activities | Medium size, real-world conditions | https://www.mdpi.com/1424-8220/22/17/6513 |
| **Student’s own phone** | Use smartphone sensor logs (see "Your Phone as a Sensor" below) | Variable, highly engaging | See section below |
| **Built-in datasets** | `scikit-learn` digits, `make_classification`, `make_moons` | Tiny, no download needed | `pip install scikit-learn` |

---

## Using Students’ Own Phone Data — Yes, absolutely!

Students can record accelerometer/gyroscope data from their own phones and use it in the exercises. Here are three practical approaches:

### Approach A: Python on Mobile
- **iOS**: Pythonista app (paid) or `objc_util` to read CoreMotion.
- **Android**: Termux + `python` + `jnius` to access accelerometer.
- **Both**: Record data CSVs via sensor‑logging apps (e.g., "Accelerometer Graph" free apps) and email/share the CSV.

### Approach B: Off‑the‑shelf sensor‑logging apps
| Platform | App (free/low‑cost) | What it exports |
|---|---|---|
| iOS | "Accelerometer Data Logger" | CSV with timestamp, x, y, z |
| Android | "Sensor Logger" (by RNProjects) | CSV with x, y, z, activity label |
| Both | "Smart Logger" | Custom labels, email CSV |

Students record **3–5 minutes** of a chosen activity (walking, standing, sitting, climbing stairs), email the CSV, and then place it in the project folder.

### Approach C: Use the `FremontBridge` / `BicycleWeather` data with a "my‑activity" twist
Even without phone sensors, students can frame the existing FremontBridge or BicycleWeather data as "counting human activity" and add their own simple observations (e.g., "I walked for 20 min at 10 am, here’s my step count").

---

## Can part of the project be a Mobile App?

**Yes.** You can assign a small mobile‑app component that connects to the same analysis pipeline. Two low‑friction options:

### Option 1: Create a "data collector" app (not a full ML model)
- The app simply records sensor readings + a user‑entered activity label, exports a CSV.
- Students then take the CSV and run the Python notebooks from the course.
- **Tech stack**: 
  - iOS: Swift with `CoreMotion` → write CSV to Documents.
  - Android: Kotlin + `SensorManager` → write CSV.
  - Multi‑platform: **Flutter** or **React Native** with sensor plugins (`accelerometer`, `gyroscope`).
- **Why it works**: The heavy ML work stays in the Python notebooks; the app is just a data‑capture wrapper.

### Option 2: Tiny ML model on the device (advanced)
- Export a tiny decision‑tree or k‑Nearest-Neighbors model (trained on the UCI data) and embed it in a mobile app that classifies the **real‑time** sensor stream.
- **Frameworks**: Apple CoreML, Android TensorFlow Lite Lite.
- **Note**: This adds complexity; recommend only if the course already covers mobile ML.

---

## List of 6 Exercises

### Exercise 1‑1: kNN on Simple Activities (Foundations)

**Focus**: kNN, feature scaling, choice of k.

**Data**: UCI minimal accelerometer (accX‑accZ + gyroX‑gyroZ, label: 1=standing, 2=walking).

**Steps**:
1. Load the data, drop incomplete rows, report activity distribution.
2. Run kNN **without** scaling and **with** `StandardScaler`; report 5‑fold CV accuracy for each.
3. Interpret the scaling effect: why does raw acceleration need scaling?
4. For k = 1, 3, 5, 10, 20 plot accuracy and comment on the curve.
5. Plot the confusion matrix and identify which activities are most confused.

**Expected takeaway**: Standing vs. walking are easily confused without scaling; k‑large reduces variance but may underfit.

---

### Exercise 1‑2: PCA on Sensor Data (Dimensionality Reduction)

**Focus**: PCA, explained variance, projection.

**Data**: Same minimal dataset (6 features).

**Steps**:
1. Fit PCA, plot explained variance ratio for each component.
2. Project the 6‑D data onto the first two principal components and visualize the two activities.
3. Identify which original features (acc peaks, gyro magnitude) contribute most to the first two components.
4. Retrain kNN using only the first 2 PCA components and compare accuracy to the full‑6‑D version.
5. Report how many components are needed to retain ≥ 95 % variance.

**Expected takeaway**: The first 2 PCs already separate standing and walking well; PCA reduces noise and improves generalization.

---

### Exercise 2‑1: MLE vs MAP with Limited Data (Bayesian & Statistical)

**Focus**: Maximum Likelihood Estimation, MAP with priors, effect of data scarcity.

**Data**: Short sequences (e.g., 100 samples) from each activity of the minimal dataset.

**Steps**:
1. Compute the MLE activity probabilities: `p(i) = count(i) / total`.
2. Assume a Beta prior (e.g., α=β=1 for uniform, or α=β=2 for a mildly informative prior) and compute the MAP estimate for each activity’s probability.
3. Compare MLE vs MAP when you have:
   - 100 samples (plenty),
   - 10 samples (scarce),
   - 2 samples (very scarce).
4. Show how MAP "smooths" the estimates toward the prior when data are few.
5. Write a short analysis (≤ 150 words) of when a Bayesian approach is preferable in HAR.

**Expected takeaway**: MAP prevents over‑confident probability estimates when training data are limited—a common real‑world HAR situation.

---

### Exercise 2‑2: Small MLP on Time‑Series Activity (Linear Models & Neural Networks)

**Focus**: MLP, forward pass, PyTorch basics.

**Data**: Normalized sequences from the minimal dataset (each sequence: 50 rows × 6 features, label = activity).

**Steps**:
1. Define a tiny MLP in PyTorch: one hidden layer with 16 neurons, ReLU activation, output with 2 logits (standing/walking).
2. Use `CrossEntropyLoss` and an optimizer (SGD or Adam).
3. Train for 5 epochs on the limited data (you may artificially augment by interpolating between consecutive time steps).
4. Plot the training loss curve.
5. Misclassification analysis: which time steps or features cause the most errors?

**Expected takeaway**: Even a tiny network can learn simple patterns from a few seconds of sensor data; overfitting is visible if epochs are increased too much.

---

### Exercise 3‑1: GMM and BIC for Number of Hidden States (Latent Variable Models)

**Focus**: Gaussian Mixture Models, EM algorithm, BIC for model selection.

**Data**: All 6 features from the minimal dataset (or just the 3 acceleration columns).

**Steps**:
1. Fit GMMs with K = 1, 2, 3, 4, 5 components (full covariance).
2. For each K, record the log‑likelihood and compute BIC = −2·logL + p·log(n) (p = #parameters, n = #samples).
3. Plot BIC vs. K and identify the "elbow" (optimal K).
4. Using the selected GMM, assign each sample the component with highest posterior probability and interpret which component corresponds to which activity.
5. Reflect: does BIC tend to select more components than AIC in this setting?

**Expected takeaway**: BIC penalizes model complexity more strongly than AIC; for HAR it often selects K = 2 (standing/walking) or K = 6 (if more activities are present).

---

### Exercise 3‑2: Hidden Markov Model on Activity Sequence (Other Topics)

**Focus**: HMM, Viterbi algorithm, hidden states.

**Data**: Short activity sequences from the minimal dataset (e.g., 80 time steps of accelerometer magnitude).

**Steps**:
1. Train a 3‑state Gaussian HMM using the Baum‑Welch (expectation‑maximization) algorithm.
2. Apply the Viterbi algorithm to find the most likely hidden‑state path for a new test sequence.
3. Compare the Viterbi‑predicted states with the true activity labels: which states correspond to standing, walking, and transition periods?
4. Experiment with transition matrix constraints (e.g., allow self‑transitions vs. force state changes every step).
5. Write a brief note on how HMM "smooths" rapid label changes compared with a plain k‑NN classifier.

**Expected takeaway**: HMM captures temporal dynamics (e.g., a walk typically lasts several seconds) and produces a denoised state sequence.

---

### Exercise 4‑1: Trees & Random Forest on Count Data (Other Topics)

**Focus**: Decision trees, random forests, leakage awareness.

**Data**: `FremontBridge.csv` (hourly pedestrian counts + weather features: temperature, rain, etc.).

**Steps**:
1. Engineer a simple feature: `hour = hour_of_day`, `is_weekend`.
2. Create a binary target: `high_count = (count > median(count))`.
3. Train a `RandomForestClassifier` with the weather features + engineered features.
4. Report `feature_importances_` and discuss which variables matter most.
5. **Crucial**: Briefly discuss leakage—if the target for hour t+1 uses features from hour t (or future data), accuracy will be artificially high. Demonstrate with a hold‑out split or 5‑fold CV and compare the results.

**Expected takeaway**: Feature engineering (hour‑of‑day, weekend flag) is essential; CV prevents optimistic bias from leakage.

---

### Exercise 4‑2: Synthesis Table: Which Technique Covers Which Course Topic? (Cross‑Course Synthesis)

**Focus**: Integration, mapping course modules to HAR techniques.

**Task**: Create a 6 × 6 table (rows = techniques, columns = course topics). For each cell write:
- A one‑sentence example usage (e.g., "kNN on scaled accelerations").
- Whether the technique is **directly applicable** to the final HAR project.
- Any data‑or time‑series requirements.

**Techniques (rows)**: kNN, PCA, MLE/MAP, MLP, GMM, HMM.
**Course topics (columns)**: Foundations, Dimensionality Reduction, Bayesian & Statistical, Neural Networks, Latent Variable Models, Other Topics.

**Expected takeaway**: Students see how all course pieces fit together and can decide which techniques to reuse in their final project.

---

## How to Get Started

1. **Set up a Python environment** (recommended):
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   pip install numpy pandas scikit-learn matplotlib ucipmlrepo
   ```
2. **Download the minimal dataset** (2 MB):
   - Using `ucimlrepo`: `pip install ucimlrepo; from ucimlrepo import fetch_ucirepo; har = fetch_ucirepo(id=755)`.
   - Or manually download the CSV from https://archive.ics.uci.edu/static/public/755/accelerometer_gyro_mobile_phone_dataset.csv and place it in `notebooks/HWs/`.
3. **Run the first exercise** (1‑1) to confirm the pipeline works.
4. **Optional: students bring their own phone data** — have them log 3‑minute walks/stands using a sensor‑logger app, email the CSV, and drop it in the project folder.

---

## Connection to the Final HAR Project

Each exercise pre‑builds one (or more) of the five final‑project sections:

| Exercise | Final‑Project Section |
|---|---|
| 1‑1 | §1 — PCA + Mahalanobis distance |
| 1‑2 | §1 (feature reduction) |
| 2‑1 | §2 — Three‑way comparison (NB, linear, kernel) |
| 2‑2 | §2 (neural‑network baseline) |
| 3‑1 | §3 — GMM & BIC for model order |
| 3‑2 | §4 — Sequence memory / HMM |
| 4‑2 | Overview / synthesis of the whole course |

The exercises are independent, but their outputs (plots, trained models, BIC curves, Viterbi paths) can be directly reused in the final project report.

---

## Grading / Evaluation (per exercise)

- **Technical correctness** (40 %): code runs without errors, correct results/plots.
- **Analysis** (40 %): short written answer interpreting the results.
- **Presentation** (20 %): notebook is readable, clear headings, concise conclusions.

**Deadline**: 1 week per exercise; late submissions lose 10 % per day (max 3 days late).

---

> **Key point**: The minimal UCI dataset (2 MB, 2 activities, 6 features) lets every exercise run in < 30 seconds on a laptop, so students can complete all 6 in a single lab session or spread them over the term. If internet is limited, the same exercises can be run on the project’s built‑in data (`FremontBridge.csv`, `BicycleWeather.csv`) with the "activity‑framing" modifications noted in the table above.