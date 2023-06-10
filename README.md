# Quantum-Optical-ConvNet
A repository for the Research Project Constructing a Quantum Optical Convolutional Neural Network (QOCNN) and providing scripts that evaluate future feasibility.

## Edits in this particular fork

In this fork we perform cat vs dog classification with QOCNN. The data is placed under 'Data/catdog'

![cat](https://github.com/SubrataSarkar32/Quantum-Optical-ConvNet/blob/master/Data/catdog/test/gatto/1.jpeg?raw=true)    VS   ![dog](https://raw.githubusercontent.com/SubrataSarkar32/Quantum-Optical-ConvNet/master/Data/catdog/test/cane/OIP--9pxEn5HUPsqA38ao3O7TgHaGV.jpeg)

To run this on a different dataset just change the directory paths in 'mnist.py' to your data directory , note images are turned into grayscale and resized to 28x28 and then classification is performed

## Requirements

Runs much faster with GPU with CUDA cores.

## Credits

The codebase here is based on https://github.com/rishab-partha/Quantum-Optical-ConvNet which is based on https://github.com/mike-fang/imprecise_optical_neural_network. The CNN code was provided by Rohan Bhomwik. Modifications are contained within the subfolder trained_models, and significant parts of mnist.py, optical_nn.py, and train_mnist.py have been written by ![Riashab Parthasarathy](https://github.com/rishab-partha). In addition, the subfolder Data and the files ROC Curves.ipynb, t1-t10.pt, and y1-y10.pt are completely new. The dataset loader and test dataset loader have been modified by ![Subrata Sarkar](https://github.com/SubrataSarkar32).

If you wish to train/load/run a sample QOCNN, write 'python train_mnist.py' in the command line and execute.

If you wish to load a sample ONN, write 'python mnist.py' in the command line and execute (WARNING: This may take a while.).

Finally, if you wish to run through the process of generating the ROC curves for a sample QOCNN, run through the file 'ROC Curves.ipynb'.

## Donation

If this fork of QOCNN helped you you save your time. You can give me a cup of coffee. :)

You can donate via BHIM UPI


![Super sub](https://github.com/SubrataSarkar32/subratasarkar32.github.io/blob/master/images/Supersub(200x200).jpg?raw=true)


[![Donate](https://github.com/SubrataSarkar32/subratasarkar32.github.io/blob/master/images/bhimupi(100x15).jpg?raw=true)](upi://pay?pn=Subrata%20Sarakar&pa=9002824700%40upi&tn=Donation&am=&cu=INR&url=http%3A%2F%2Fupi.link%2F)

Scan this QR Code to open your UPI Payment App on your phone

![QR code](https://github.com/SubrataSarkar32/subratasarkar32.github.io/blob/master/images/qrpay.png?raw=true)
