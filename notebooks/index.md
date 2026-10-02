# Machine Learning

**Mahmood Amintoosi, Fall 2026**

Computer Science Dept, Ferdowsi University of Mashhad

Required, 4 credits — M.Sc. Computer Science, Soft Computing and Artificial Intelligence track.

---

# About this course

This course gives an **integrated picture of machine learning foundations**, **hypothesis and model evaluation**, and the use of methods on problems of **moderate complexity**. It is designed for new M.Sc. students from mixed undergraduate backgrounds: we start from shared concepts and standard evaluation, then move from simple models to more advanced ones, with enough mathematics to use classical methods carefully.

**Learning paradigms:** supervised, unsupervised, and semi-supervised learning.  
**Tasks:** classification, regression, clustering, and ranking.  
**Themes:** overfitting and underfitting, the bias–variance tradeoff, the curse of dimensionality, and principled model comparison.

Topics progress roughly as: local models (kNN) and k-means → high-dimensional data and dimensionality reduction (PCA, LDA) → linear models and gradient methods → neural networks (MLP, backpropagation) → kernel methods → Bayesian learning and decision theory → Gaussian mixtures and EM → Decision trees → learning to rank and learning theory → advanced topics and a project.

<!-- **Other courses in the track** cover support vector machines, reinforcement learning, and ensembles / model combination. Related material also lives in [Neural Networks](https://fum-cs.github.io/neural-networks/), [Statistical Machine Learning](https://fum-cs.github.io/SML/), [Algorithms for Data Science](https://fum-cs.github.io/a4ds/), and [MFDS](https://fum-cs.github.io/mfds/). -->

The Spring 2025 snapshot of this book is tagged [`2025-spring`](https://github.com/fum-cs/machine-learning/tree/2025-spring).

## Learning outcomes

By the end of the course, you should be able to:

- Analyze learning paradigms and problem types (classification, regression, clustering, ranking);
- Explain overfitting, underfitting, bias–variance, and the curse of dimensionality, and relate them to model choice;
- Understand and implement kNN, k-means, linear models and gradient methods, MLP and backpropagation, PCA/LDA, kernel methods, MLE / Naive Bayes / Bayesian learning, and EM / Gaussian mixtures;
- Evaluate hypotheses and models with standard criteria and interpret the results;
- Read, analyze, and present a related scientific paper;
- For a real problem: prepare data, select and implement methods, compare them, and write a short scientific report.

## Prerequisites

- Python programming and problem solving  
- Probability and statistics  
- Linear algebra  
- Mathematics needed for ML (differentiation, introductory optimization)  

An introductory review of some material is included in the course. You do not need to have taken a full ML course before, but basic familiarity with the vocabulary helps.

## Student work

Pre-class reading, class discussion, theoretical and practical exercises, algorithm implementation (Python / scikit-learn, and from scratch when needed), **student seminars** (learning to rank; computational learning theory and VC dimension), and an **applied project** (data preparation, comparison of several models, evaluation and report).

## AI tools policy

AI tools are **allowed but not a substitute for your own understanding**. You may use them to learn concepts, debug code, brainstorm, review code, and improve writing. You may **not** present AI-generated content as your own independent work without reviewing and understanding it. You are responsible for scientific accuracy, code, results, and sources; in assessed work you may be asked to explain your process and technical decisions.

## Assessment

Quizzes during the term and a final exam · assignments · project · student presentations/seminars · active participation in class.

## Teaching method

Course notes and Jupyter notebooks in this book (GitHub), programming in Python and scikit-learn; in-class teaching and troubleshooting with discussion, Q&A, worked examples, and student presentations. Teaching is in person, with use of the VU system where needed.

## Questions?

Office hours: Sunday 8:00–9:30, or email m.amintoosi@um.ac.ir, or talk after class, or [book a slot](https://calendly.com/m-amintoosi/30min).

## Slack

Join the [FUM CS Slack](https://join.slack.com/t/fum-cs/shared_invite/zt-1zntzuw2t-JOWbsyQdGASNz~40AhWy_Q) for course discussion.

---

## Textbooks

1. M. J. Zaki and W. Meira Jr., *Data Mining and Machine Learning: Fundamental Concepts and Algorithms*. Cambridge University Press, 2020.  
2. R. O. Duda, P. E. Hart, and D. G. Stork, *Pattern Classification*, 2nd ed. John Wiley & Sons, 2001.  
3. T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. Springer, 2009.  
4. C. M. Bishop, *Pattern Recognition and Machine Learning*. Springer, 2006.  
5. K. P. Murphy, *Machine Learning: A Probabilistic Perspective*. MIT Press, 2012.  
6. S. Theodoridis, *Machine Learning: A Bayesian and Optimization Perspective*, 2nd ed. Academic Press, 2020.  
7. J. VanderPlas, *Python Data Science Handbook*. O’Reilly Media, 2016. [Jupyter notebooks on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook)

## Related papers

In addition to the textbooks, selected research and survey papers will be used—on bias–variance and model complexity, the curse of dimensionality, dimensionality reduction (PCA/LDA), kernel methods, learning to rank, and computational learning theory (including VC dimension). For seminar sessions, a short reading list will be provided.

```{bibliography}
```

---

*Part of the early material is adapted from the [Open Machine Learning Course](https://ml-course.github.io/) by Joaquin Vanschoren and others. Later chapters reuse and adapt notebooks from related FUM courses; we thank the original authors.*
