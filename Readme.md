
# Football Player and Element Detection with Deep Learning

## Project Overview

This project uses deep learning techniques to detect football players, identify their team affiliation (based on uniform colors), and recognize other elements such as the ball and the referee in any given image. The model is trained on a dataset of football images to distinguish between players, teams, referees, and the ball.

The project includes scripts for training the model, testing it on new data, and running the full end-to-end workflow. Below is an explanation of the project structure and instructions for running it.

---

## Project Structure
***** Directory: Structure *****

- Train.py
- Test.py
- Run.py (main)
- Dataset.py

- Data/
	- Train/
		- proj_det
		- proj_gt
		- proj_img1
	- Test/
		- proj_det
		- proj_gt
		- proj_img1
- trained_models/

***** Running *****

(a) Run Train.py to train the model. -> Generates a new trained model.
(b) Run Test.py to test the model.
(c) Run Run.py to run the full project (both Train and Test)
