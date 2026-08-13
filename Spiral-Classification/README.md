Classification Problems
GOAL:

To implement and compare different classification methods on the 2-spiral, 3-spiral and BUPA classification problems.

Artificial Neural Networks (ANNs) and Support Vector Machines (SVMs) were used to solve the classification problems.

ChatGPT was also given the classification problems to attempt independently. These attempts were kept in separate notebooks so they could be compared to the manually implemented solutions.

Models:

2-Spiral:

Artificial Neural Network
Support Vector Machine

3-Spiral:

Artificial Neural Network
Support Vector Machine

BUPA:

Classification methods were tested on the BUPA liver disorder dataset.
2-Spiral Classification:

The 2-spiral problem contains two classes arranged into two intertwined spirals. This makes the dataset difficult to classify using a simple linear classifier.

An ANN was used to learn the non-linear boundary between the two classes.

An SVM was also used to classify the dataset and compare its performance against the neural network.

Relevant files:

Two_spiral.ipynb
Two_SpiralSVM.ipynb
spiralsdataset.csv
3-Spiral Classification:

The 3-spiral problem extends the original problem by introducing a third intertwined class, increasing the difficulty of the classification task.

Both an ANN and SVM were used to attempt the classification problem.

Relevant files:

Three_spiral.ipynb
3-Spiral SVM.ipynb
Threespirals.csv
BUPA Classification:

The BUPA dataset was also used to test classification methods on real-world data rather than the artificially generated spiral datasets.

Relevant files:

Bupa.ipynb
bupa.csv
ChatGPT Attempts:

ChatGPT was also used to attempt the classification problems.

The ChatGPT-generated code was kept separate from the manually written implementations so the different approaches could be tested and compared.

The ChatGPT notebooks include:

BUPA ChatGPT.ipynb
BUPA Testing.ipynb
Failed Dichotomization ChatGPT.ipynb
3-spiralSVM.ipynb
Observations:

Both ANNs and SVMs were capable of handling non-linear classification problems.

The 3-spiral problem was more difficult than the 2-spiral problem due to the additional class and more complex decision boundaries.

The BUPA dataset provided a different classification problem using real-world data instead of generated spiral data.

The ChatGPT attempts also allowed the AI-generated approaches to be compared against the manually implemented solutions.

Technologies:

Python
Jupyter Notebook
PyTorch
Scikit-learn
Pandas
NumPy
Matplotlib

Author

Patrick Triggell C3332224