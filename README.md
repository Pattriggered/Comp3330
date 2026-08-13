Food Classification CNN
GOAL:
to implement a CNN for classifying 40 food groups using the image data, The data was split into
training validation and test sets. The inference file will only run the ResNet\_50 model as it was
the highest performing of the three models. However, all models are available as notebooks.

Model:
ResNet18, ResNet50 (Patrick Triggell C3332224)
mobileNetV2 (Sharwil Purohit C3484094)
classes:40
input size: 224 x 224

How to run inference:
through the terminal
python inference.py <dataset\_folder>
Example: cd C:\\MOAD (Correct directory)
python inference.py MOAD\_Split\\test

Running additional code:
All code (Excluding inference.py) were written and well documented in
Jupyter notebooks, they can be run using any compatible IDE.
The "Preprocessing" file is specifically for preparation of the data before the training process,
both CNN files are the training of their respective models,

Results:
Mean Class Accuracy: 83.64%
Overall Accuracy 86.05%

Observations:
strong performance on visually distinct classes
lower accuracy on fish and meats (possibly due to smaller sample sizer or similarities between
classes)

Future improvements:
Increase dataset size, specifically for underperforming classes
Use class balancing techniques
slightly more fine tuning with data augmentation

References:
Images sourced from Pexels and Unsplash
Usage of ResNet18, ResNet50 and MobileNetV2 for transfer learning



\#Author
Patrick Triggell C3332224

Sharwil Purohit C3484094

