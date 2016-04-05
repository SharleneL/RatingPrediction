# Rating Prediction Source Code
**Author: Shalin Luo**

- To run the code, change the parameters in `main.py` and run `main.py` file.
- In this project, I use a strategy to save some middle helper files. The reason to do this is to save the time to do the data preprocess and feature extraction, when the feature matrix are reusable.
	- In `main.py` file's **GENERATE MIDDLE FILES** part(line 30 - 40), the code will generate a libsvm format file for both the training set and dev/test set's feature matrix, as well as a file containing the feature list with ctf/df count. 
	- In `main.py` file's `USE MIDDLE FILE` part (line 43 - 51), the code will read in from the hard coded file path and generate needed matrix quickly.
	- Thus, when rerun the code with the same matrix, you can just comment **GENERATE MIDDLE FILES** part(line 30 - 40) and uncomment `USE MIDDLE FILE` part (line 43 - 51), then run `main.py`.