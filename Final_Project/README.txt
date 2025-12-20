README

NOTE: The dataset used by the RNN is too large to host on GitHub. The trained RNN (TRAINED_RNN.pth) is availible though. 

Required packages: panda, numpy, torch, tqdm, math. Model was developed with Python 3.12 but should work fine with the latest versions through 2025. 

To train the model, simply download the files Alden_Erazo_RecurrentNeuralNetwork_PhishingDetector.ipynb and CombinedDataSetWithEmailMaximum512Tokens.csv. They should be saved in the same folder. Then you can open the .ipynb and run all cells. Output should appear below the lowest cell with Python code. The dataset is split with a set seed but the model is trained on it in a random order, so final accuracy may differ very slightly each time you run it.

If you want to run a version of the model that has been trained, use the TRAINED_RNN.pth file. It had 97% accuracy in testing. 
