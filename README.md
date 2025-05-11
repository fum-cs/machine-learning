![](lectures/img/572_banner.png)

## Computer Science Dept, Ferdowsi University of Mashhad

# Machine Learning

An introduction to machine learning.

- [Course Jupyter Book](https://fum-cs.github.io/machine-learning/README.html)

2024 Instructor: Mahmood Amintoosi

I should mention that the original material of this course was from [Open Machine Learning Course](https://ml-course.github.io/), by [Joaquin Vanschoren](https://github.com/joaquinvanschoren) and others.

### ML-related textbooks

- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems by Aurélien Géron. Code/notebooks available [here](https://github.com/ageron/handson-ml2). (Endorsed by an MDS student!)
- James, Gareth; Witten, Daniela; Hastie, Trevor; and Tibshirani, Robert. An Introduction to Statistical Learning: with Applications in R. 2014. Plus [Python code](https://github.com/JWarmenhoven/ISLR-python) and [more Python code](https://github.com/mscaudill/IntroStatLearn).
- Russell, Stuart, and Peter Norvig. Artificial intelligence: a modern approach. 1995.
- David Poole and Alan Mackwordth. Artificial Intelligence: foundations of computational agents. 2nd edition (2017). [Free e-book](http://artint.info/).
- Kevin Murphy. Machine Learning: A Probabilistic Perspective. 2012.
- Christopher Bishop. Pattern Recognition and Machine Learning. 2007.
- Pang-Ning Tan, Michael Steinbach, Vipin Kumar. Introduction to Data Mining. 2005.
- Mining of Massive Datasets. Jure Leskovec, Anand Rajaraman, Jeffrey David Ullman. 2nd ed, 2014.

### Math for ML

- [Mathematics for Machine Learning](https://mml-book.github.io/)
- [The Matrix Calculus You Need For Deep Learning](http://parrt.cs.usfca.edu/doc/matrix-calculus/index.html)
- [Introduction to Optimizers](https://blog.algorithmia.com/introduction-to-optimizers/)

### Other ML resources

- [A Course in Machine Learning](http://ciml.info/)
- [Nando de Freitas lecture videos](https://www.youtube.com/watch?v=PlhFWT7vAEw) and [online course](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/)

### Interesting ML Competition Write-Ups

- [Diabetic retinopathy Kaggle competition write-up](http://jeffreydf.github.io/diabetic-retinopathy-detection/)
- [Galaxy Zoo Kaggle competition write-up](https://benanne.github.io/2014/04/05/galaxy-zoo.html)
- [National Data Science Bowl competition write-up](https://benanne.github.io/2015/03/17/plankton.html)


## Build

- jupyter-book build ./
- ghp-import -n -p -f ./_build/html
- jupyter-book build --builder pdflatex ./
