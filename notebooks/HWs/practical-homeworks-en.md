# Practical Homeworks — English version

*.objective*: A set of lightweight practical assignments (≈1‑3 h each) at the end of each lecture topic, plus a set of five Human‑Activity‑Recognition (HAR)‑themed topics that lead into the course’s final project. Designed so that later students can run the notebooks, reproduce the results, and use the assignments as building blocks for the final project.

---

## General Rules

- Each homework is designed for **about 1‑3 hours** of work. The goal is to reinforce the concepts of the corresponding lecture, not to achieve the highest possible accuracy.
- Homeworks are **individual**: discussion of ideas is encouraged, but the code and write‑up must be your own.
- **Submission**: a Jupyter notebook that runs from top to bottom without errors **plus** a short analysis (max 1 page, may be Markdown cells at the end of the notebook).
- Allowed libraries: `scikit‑learn`, `pandas`, `matplotlib`, `PyTorch`. Use of AI tools is permitted per course policy, but you must be able to explain every line of code.
- Each homework is assessed on three aspects (see the “Grading” section at the end).

---

## Homework Map (Chapter → Assignments)

| Chapter (as in the book) | Homework(s) | Main skill |
|---|---|---|
| **Foundations** | 1‑1, 1‑2 | kNN, feature scaling, k‑means |
| **Simple Methods & High Dimension** | 2‑1, 2‑2 | k‑means from scratch, curse of dimensionality |
| **Linear Models & Neural Networks** | 3‑1, 3‑2 | Gradient descent vs. small MLP |
| **Dimensionality Reduction** | 4‑1, 4‑2 | PCA, PCA vs. LDA |
| **Kernel Methods** | 5‑1, 5‑2 | Feature map, Nadaraya‑Watson kernel regression |
| **Bayesian & Statistical Learning** | 6‑1, 6‑2 | MLE vs. MAP, Naïve Bayes |
| **Latent Variable Models** | 7‑1 | GMM, EM, BIC |
| **Other Topics** | 8‑1, 8‑2 | Trees / Random Forest, Viterbi / HMM |

*Total: 15 chapter‑specific homeworks + 5 HAR‑themed topics.*

---

## Chapter‑by‑Chapter Homework Summaries (English)

### Foundations
**1‑1 kNN on Penguin Features**  
- **Data**: Palmer Penguins (seaborn).  
- **Tasks**: scale features, train kNN with and without scaling, plot accuracy for k = 1,3,5,9,15,25,45, inspect confusion matrix, visualise decision boundaries for k=1 and k=15.  
- **Deliverable**: notebook + a short paragraph “Why did scaling have such a large effect? Why isn’t ‘big‑k’ always better?”

**1‑2 Custom Colour Palette with k‑means**  
- **Data**: Any image (or `skimage.data.astronaut()`).  
- **Tasks**: flatten image pixels, run k‑means (K = 16) to reconstruct the image, visualise the resulting palette, compute compactness (inertia).  
- **Deliverable**: notebook + a brief analysis of how K affects visual quality.

### Simple Methods & High Dimension
**2‑1 k‑means from Scratch**  
- **Data**: `sklearn.datasets.load_digits()` (or the ` digits` matrix).  
- **Tasks**: implement Lloyd’s algorithm (initialise centroids, assign, recompute), run until convergence, evaluate with Adjusted Mutual Information against the true labels.  
- **Deliverable**: notebook + discussion of convergence behaviour vs. number of centroids.

**2‑2 Curse of Dimensionality**  
- **Data**: `sklearn.datasets.make_classification` or the ` digits` dataset.  
- **Tasks**: add random noise features (0, 10, 50, 100, 300, 600), train a kNN classifier each time, plot test accuracy.  
- **Deliverable**: notebook + interpretation of the drop (or lack thereof) in accuracy.

### Linear Models & Neural Networks
**3‑1 Gradient Descent on Fremont Bridge counts**  
- **Data**: `FremontBridge.csv` (hourly pedestrian counts + weather).  
- **Tasks**: implement batch gradient descent for simple linear regression predicting count from hour‑of‑day; plot loss curve; compare with `sklearn.linear_model.LinearRegression`.  
- **Deliverable**: notebook + a short note on convergence speed and step‑size choice.

**3‑2 Small MLP in PyTorch**  
- **Data**: `FremontBridge.csv` or `BicycleWeather.csv` (binary classification: “high count” vs “low count”).  
- **Tasks**: define a one‑hidden‑layer MLP (16 hidden units, ReLU), train with cross‑entropy for 5 epochs (you may artificially augment data), plot loss, discuss over‑/under‑fitting.  
- **Deliverable**: notebook + analysis of how the tiny network behaves on the data.

### Dimensionality Reduction
**4‑1 Eigen‑digits**  
- **Data**: `sklearn.datasets.load_digits()`.  
- **Tasks**: compute the eigen‑faces (principal components of the covariance matrix), reconstruct digits using 1, 5, 10, 20 components, plot original vs. reconstruction, measure reconstruction error.  
- **Deliverable**: notebook + explanation of how increasing components improves fidelity.

**4‑2 PCA vs. LDA**  
- **Data**: `digits` dataset (or `make_classification` with two clusters).  
- **Tasks**: fit PCA (project to 2D) and LDA (maximise class separation), visualise the 2‑D projections, comment on why LDA can separate classes better when within‑class variance differs across classes.  
- **Deliverable**: notebook + a paragraph on the assumptions each method makes.

### Kernel Methods
**5‑1 Two Moons, Explicit Feature Map**  
- **Data**: `sklearn.datasets.make_moons()`.  
- **Tasks**: generate the two‑moon dataset, train a linear SVM in the explicit feature space (use `sklearn.preprocessing.PolynomialFeatures` degree 2), compare accuracy with an RBF‑kernel SVM.  
- **Deliverable**: notebook + discussion of the trade‑off between explicit map size and kernel trick.

**5‑2 Nadaraya‑Watson Kernel Regression**  
- **Data**: any simple 1‑D regression set (e.g., `y = sin(x) + noise`).  
- **Tasks**: implement the Nadaraya‑Watson estimator with a Gaussian kernel and varying bandwidths `h`; plot fits; select `h` via leave‑one‑out cross‑validation.  
- **Deliverable**: notebook + commentary on bias‑variance trade‑off via bandwidth.

### Bayesian & Statistical Learning
**6‑1 MLE vs. MAP with Little Data**  
- **Data**: synthetic counts from a Bernoulli process (e.g., 10, 20, 50 samples).  
- **Tasks**: compute MLE for the success probability, then compute MAP with a Beta(α,β) prior (try α=β=1 uniform, and α=β=2 informative). Plot the two estimates side‑by‑side for each sample size.  
- **Deliverable**: notebook + short analysis of when MAP “protects” us from over‑confident MLE.

**6‑2 Naïve Bayes on SMS Spam**  
- **Data**: `UCI SMS Spam Collection` (or the built‑in `datasets` if available).  
- **Tasks**: fit a multivariate Bernoulli Naïve Bayes classifier, report accuracy, confusion matrix, and the top 10 most informative words (features).  
- **Deliverable**: notebook + reflection on the “naïve” independence assumption.

### Latent Variable Models
**7‑1 GMM & BIC on Moons**  
- **Data**: `sklearn.datasets.make_moons()`.  
- **Tasks**: fit GMMs with K = 1…6 components (full covariance), compute BIC for each K, plot BIC curve, select the K with minimum BIC, visualise the resulting clusters.  
- **Deliverable**: notebook + commentary on how BIC balances fit vs. complexity.

### Other Topics
**8‑1 Tree / Forest on Fremont Bridge with Leakage Check**  
- **Data**: `FremontBridge.csv`.  
- **Tasks**: engineer features (hour‑of‑day, weekend flag), create a binary target “high traffic”, train a `RandomForestClassifier`, report feature importances, then discuss potential leakage if target for hour t uses features from hour t+1; verify with 5‑fold CV.  
- **Deliverable**: notebook + note on leakage and how CV mitigates it.

**8‑2 Viterbi on a Real Fremont Sequence**  
- **Data**: a short segment of `FremontBridge.csv` (e.g., 80 consecutive hourly counts).  
- **Tasks**: treat the counts as observations of a 3‑state HMM (states: “low”, “medium”, “high”), train with Baum‑Welch, run Viterbi to obtain the most likely hidden‑state path, compare the inferred states with simple thresholds.  
- **Deliverable**: notebook + brief note on what the hidden states represent.

---

## Five HAR‑Themed Topics (leading to the Final Project)

These topics are **not** tied to a specific lecture week; they are meant to be worked on (individually or in groups) after the core content, and their outputs can be directly reused in the course’s final Human‑Activity‑Recognition project.

| Topic | Relation to Final Project |
|---|---|
| **1. PCA + Mahalanobis distance** | §1 of the final project – feature reduction + distance‑based classification. |
| **2. Three‑way comparison (Naïve Bayes / Linear kernel / RBF kernel)** | §2 – baseline classifiers on the HAR dataset. |
| **3. GMM & BIC for model order** | §3 – selecting the number of Gaussian components. |
| **4. Sequence memory / HMM (Viterbi)** | §4 – modelling temporal dependencies in activity sequences. |
| **5. From numbers to a report / reproducibility** | §5 – writing the final project report, ensuring all steps are reproducible from the notebooks. |

Each topic provides a short notebook (≈30‑45 min) that produces a artefact (plot, trained model, BIC curve, Viterbi path, or a short analysis paragraph) that can be copied into the final project write‑up.

---

## Grading / Evaluation (per homework)

| Criterion | Weight | Description |
|---|---|---|
| **Technical correctness** | 40 % | Code runs without errors; produces the expected plots / numbers. |
| **Analysis** | 40 % | Short written answer (≤ 200 words) interpreting the results, discussing trade‑offs, or explaining a chosen parameter. |
| **Presentation** | 20 % | Notebook is well‑structured, clear headings, concise markdown cells, reproducible cell order. |

**Deadline**: 1 week after assignment; late submissions lose 10 % per day (maximum 3 days late).

---

## How to Get Started

1. **Create a virtual environment** (recommended):  
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```
2. **Download the minimal UCI HAR dataset** (2 MB, 2 activities, 6 sensors) – see the English HAR‑exercises file for the exact link, or use the project’s built‑in `FremontBridge.csv` / `BicycleWeather.csv` with the “activity‑framing” modifications described there.
3. **Run the first homework** (e.g., 1‑1) to confirm the pipeline works.
4. **Optional**: have students log their own phone sensor data using a free app (see the English HAR‑exercises file for instructions) and drop the CSV into the project folder.

---

## Connection to the Final Project

Each chapter homework pre‑builds one (or more) of the five final‑project sections, and the five HAR‑themed topics directly supply artefacts for those sections. By the end of the term, students will have:

- Trained/ evaluated several classifiers (kNN, Naïve Bayes, MLP, GMM, HMM).  
- Produced PCA/LDA projections, BIC curves, Viterbi state paths.  
- Written short analysis paragraphs that can be copied verbatim into the final report.  

Thus the homeworks are **not busy‑work** – they are the building blocks of the final project.

--- 

*End of file.*