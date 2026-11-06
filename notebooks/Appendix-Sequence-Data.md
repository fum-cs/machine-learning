# Sequence Data: From Independent Samples to Ordered Observations

## 1. Why sequence data are different

Many introductory machine-learning examples represent a dataset as a table: each row is one observation, and the order of the rows is usually irrelevant. For example, in a standard classification problem, shuffling the rows does not change the meaning of the dataset.

Sequence data are different. Each observation belongs to an ordered stream, and the order may carry information. Examples include:

- accelerometer readings collected by a smartphone;
- audio samples in a speech signal;
- words in a sentence;
- measurements collected from a patient over time;
- frames in a video.

A sequence is therefore not just a collection of observations. It is an **ordered collection**, often accompanied by timestamps and boundaries that identify where one sequence starts and ends.

## 2. A sensor record is not yet a sequence model

In the WISDM v1.1 raw file, one record contains six fields:

`user, activity, timestamp, x, y, z`

The three acceleration values describe one sensor reading. The timestamp indicates when that reading was recorded. A succession of records from the same participant forms a time series.

The WISDM v1.1 timestamp is generally the phone's uptime expressed in nanoseconds; it should not be interpreted as a Unix date-time. The nominal sampling frequency is approximately 20 Hz, corresponding to about 50 milliseconds between consecutive readings. Actual time differences should be checked rather than assumed to be perfectly constant.

The three axes are measurements of the same physical process. They are not three consecutive time steps. At each time step, the observation is a three-dimensional vector:

[
mathbf{x}_t = [x_t, y_t, z_t]^	op.
]

A short sequence can be represented as a matrix whose rows are time steps and whose columns are sensor axes.

## 3. From a continuous stream to fixed-length windows

Many classical machine-learning algorithms expect a fixed-size feature vector for each example. A common way to use them with sensor streams is to divide each continuous segment into fixed-length windows.

In this project, the initial window configuration is:

- window length: 51 samples;
- step size: 26 samples;
- nominal sampling rate: 20 Hz.

Thus, a window covers approximately (51/20 = 2.55) seconds, and a new window starts approximately every (26/20 = 1.30) seconds.

The overlap is (51-26=25) samples, or approximately (25/51 approx 49.0\%) of the window length.

A window contains a short multivariate time series:

[
X_i =
egin{bmatrix}
x_1 & y_1 & z_1 \\
x_2 & y_2 & z_2 \\
\vdots & \vdots & \vdots \\
x_{51} & y_{51} & z_{51}
end{bmatrix}.
]

For a feature-based classifier, this matrix is usually transformed into a fixed-length feature vector. For a 1D-CNN, the window may instead be used as a short multichannel signal.

### Window boundaries matter

A window must not combine measurements from different participants. It must also not cross an activity boundary when the task assumes that each window has one activity label. Otherwise, the input contains a mixture of activities while being assigned a single label.

The same principle applies to longer sequences: preserve the boundaries between separate recordings or continuous segments. Do not concatenate unrelated segments and treat the artificial join as a real temporal transition.

## 4. Overlapping windows are not independent observations

With a 49% overlap, neighboring windows share many of the same raw sensor readings. This has two consequences:

1. Neighboring windows are strongly related; they are not independent samples.
2. A random split of windows can place nearly identical overlapping content in both training and test sets.

The second case creates **data leakage** and can produce overly optimistic test performance.

For this project, participants must be separated into training, validation, and test groups. All windows from a participant must remain in the same split. This subject-wise protocol tests whether a model can generalize to participants it did not see during training.

The split must be respected throughout preprocessing and model selection. In particular, transformations that learn parameters from data (for example, feature standardization or PCA) must be fitted using training data only, then applied to validation and test data.

## 5. Independent classification versus sequence modeling

There are two different prediction settings.

### Window-level classification

A classifier receives one window (or its feature vector) and predicts its activity:

[
hat{y}_i = f(X_i).
]

The prediction for a window is made without explicitly using the predictions or hidden states of neighboring windows. Logistic Regression, SVM, Gaussian classifiers, and a window-based CNN can be used in this setting.

### Sequence-level modeling

A sequence model uses the order of observations. It can represent regularities such as the fact that an activity often persists for several consecutive windows, or that some transitions are more plausible than others.

A Hidden Markov Model (HMM), for example, represents hidden states and observations through initial-state, transition, and emission probabilities. The Forward algorithm evaluates the probability of an observation sequence. The Viterbi algorithm finds the most likely hidden-state path for that sequence.

Sequence modeling does not automatically improve classification. Its usefulness depends on whether the data contain meaningful temporal structure and whether the model's assumptions are reasonable.

## 6. Keep the evaluation protocol clear

When evaluating sequence-based methods, report exactly what is being evaluated:

- Are predictions made independently for each window, or jointly for a complete sequence?
- Are sequence boundaries preserved?
- Are all model parameters estimated using training data only?
- Is validation data used for model selection, with test data reserved for final evaluation?
- Are metrics computed over the same set of windows and with the same label definitions?

Some sequence algorithms use future observations to estimate the state at an earlier time. Such smoothing is useful for offline analysis, but it is not the same as real-time prediction, where future observations are unavailable. State clearly which setting is used.

## 7. Questions to check your understanding

1. Why is it incorrect to treat the three accelerometer axes as three consecutive time steps?
2. What is the approximate duration of a 51-sample window at 20 Hz?
3. How many samples do two consecutive windows share when the step size is 26?
4. Why can a random split of overlapping windows lead to data leakage?
5. Why should windows not cross participant or activity boundaries?
6. What information can an HMM use that an independent window classifier does not explicitly model?
7. Why is offline smoothing not equivalent to real-time prediction?

## Further reading

- [WISDM Activity Prediction v1.1 dataset](https://www.cis.fordham.edu/wisdm/dataset.php)
- Rabiner, L. R. (1989). [A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition](https://doi.org/10.1109/5.18626). *Proceedings of the IEEE*, 77(2), 257–286.
