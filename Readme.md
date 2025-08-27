
# Football Player and Element Detection with Deep Learning

![Captura de pantalla 2024-09-09 211852](https://github.com/user-attachments/assets/6bdca30a-bd1b-4a86-a022-bc865d02efaa)


This project explores the use of **deep learning techniques** to detect football players, referees, and the ball in images and to classify players into their respective teams based on their uniforms. It was developed as part of a collaboration with **Veo Technologies** during the *02456 Deep Learning course* at the **Technical University of Denmark (DTU)**.

The work evaluates and compares different state-of-the-art architectures, including **YOLOv5 (various sizes)**, **Faster R-CNN with ResNet-50 backbone**, and a **pre-trained VGG16** model, applied to football match footage recorded by the Veo Cam 2 system.  
The goal was not only to detect and classify individual entities (players, referees, ball) but also to investigate methods for distinguishing teams via uniform colors, using both supervised and unsupervised approaches.


## ✨ Project Highlights
- Collaboration with **Veo Technologies** on football analytics.
- Comparison of **one-stage (YOLOv5)** vs **two-stage (Faster R-CNN)** object detection methods.
- Experimentation with **pre-trained CNNs (VGG16)** for feature extraction.
- **Player, ball, and referee detection** from real match images.
- **Team classification** using both:
  - Hand-labeled team datasets.
  - Automatic labeling via unsupervised clustering of uniform colors (GMM vs KNN).
- Evaluation using **SoccerNet** and **Veo Cam 2** datasets.
- Focus on challenges such as **small/occluded objects**, **illumination variation**, and **camera perspective**.


## 📂 Project Structure


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

## Run

(a) Run Train.py to train the model. -> Generates a new trained model.
(b) Run Test.py to test the model.
(c) Run Run.py to run the full project (both Train and Test)


## 📊 Results Overview

- **YOLOv5**  
  - Proved to be faster (up to **91 FPS**), making it a strong candidate for real-time applications.  
  - Slightly less accurate on small objects like the ball.  

- **Faster R-CNN ResNet-50**  
  - Achieved higher accuracy on small objects (ball).  
  - Required more computation and longer inference times.  

- **Team Classification**  
  - Worked reliably with **hand-labeled data**.  
  - **Unsupervised clustering** of uniform colors (via Gaussian Mixture Models) was promising but sensitive to lighting and background noise.  

- **Referee Detection**  
  - Remained the hardest task due to similarity with players and goalkeepers.  

---

## 📑 References

📑 [Full Project Report (PDF)](Deep_Learning_Final_Report.pdf)
- Collaboration with **Veo Technologies**  
- Course: *02456 Deep Learning*, DTU Compute  

