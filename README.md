## Computer Science Dept, Ferdowsi University of Mashhad

# Machine Learning

A four-credit required M.Sc. course in the **Soft Computing and Artificial Intelligence** track (usually first semester).

- [Course Jupyter Book](https://fum-cs.github.io/machine-learning/)

**Fall 2026** — Instructor: Mahmood Amintoosi

Past offerings are frozen with git tags — see [semesters.md](semesters.md).

## Course aim

An integrated view of **foundations of machine learning methods**, **evaluation of hypotheses and models**, and their use on problems of **moderate complexity**, with enough mathematics to apply classical methods carefully. The path is simple → hard: local models and clustering, then high dimensionality and why we reduce dimensions, then linear models and gradient methods, neural networks, dimensionality reduction (PCA/LDA), kernel methods, Bayesian learning, and latent-variable models (EM / GMM).

By the end of the course, students should be able to formulate problems in supervised, semi-supervised, and unsupervised settings; implement and compare methods; evaluate with standard criteria; and reason about overfitting, the bias–variance tradeoff, and high-dimensional data.

**Not covered here** (taught in other courses): support vector machines, reinforcement learning, and ensembles / model combination.

## Topics (outline)

1. Introduction: learning paradigms, tasks (classification, regression, clustering, ranking), overfitting  
2. Local models (kNN)  
3. Model and hypothesis evaluation  
4. Bias–variance tradeoff  
5. k-means clustering and clustering validation  
6. High-dimensional data and the curse of dimensionality  
7. Linear regression and linear models  
8. Gradient methods (GD / SGD)  
9. Multilayer neural networks and backpropagation  
10. PCA and LDA (dimensionality reduction)  
11. Kernel methods (feature maps, kernel trick, kernel regression)  
12. MLE, Naive Bayes, Bayesian learning, Bayesian decision theory  
13. Gaussian mixtures and EM  
14. Learning to rank 
15. Computational learning theory and VC dimension
16. Advanced topics survey  
17. Decision trees 
18. Project  

## Main references

1. M. J. Zaki and W. Meira Jr., *Data Mining and Machine Learning: Fundamental Concepts and Algorithms*. Cambridge University Press, 2020.  
2. R. O. Duda, P. E. Hart, and D. G. Stork, *Pattern Classification*, 2nd ed. John Wiley & Sons, 2001.  
3. T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. Springer, 2009.  
4. C. M. Bishop, *Pattern Recognition and Machine Learning*. Springer, 2006.  
5. K. P. Murphy, *Machine Learning: A Probabilistic Perspective*. MIT Press, 2012.  
6. S. Theodoridis, *Machine Learning: A Bayesian and Optimization Perspective*, 2nd ed. Academic Press, 2020.  
7. J. VanderPlas, *Python Data Science Handbook*. O’Reilly Media, 2016. [GitHub notebooks](https://github.com/jakevdp/PythonDataScienceHandbook)

Selected research and survey papers will be used for seminars (e.g. learning to rank; PAC / VC dimension).

Part of the early material is adapted from the [Open Machine Learning Course](https://ml-course.github.io/) by Joaquin Vanschoren and others; later chapters draw on notebooks from related FUM courses (Neural Networks, Statistical Machine Learning, Algorithms for Data Science, MFDS).

## Prerequisites

- Python programming and problem solving  
- Probability and statistics  
- Linear algebra  
- Differentiation and introductory optimization  

A brief review of some needed background is included in the course.

## Build

From the `notebooks` folder:

```
jupyter-book build ./
ghp-import -n -p -f ./_build/html
jupyter-book build --builder pdflatex ./
```

See also [production.md](production.md).

## Semester snapshots

At the end of each offering, tag the final state (do not rename this repo):

```
git tag -a YYYY-term -m "Snapshot: ..."
git push origin YYYY-term
```

Then add a row in [semesters.md](semesters.md).
