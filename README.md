## COS429 Final Project: Predicting the String Played by a Cellist through Edge Detectors, Hough Transforms, and k-Nearest Neighbors

> Final project for COS429: Computer Vision. We aim to predict the string that a cellist is playing on based on only visual data using feature detection and a k-nearest neighbors model. For more details, see final_report.pdf
---

## Table of Contents

- [Installation](#installation)
- [Dataset](#Dataset)
- [Model](#Model)
- [Evaluation](#Evaluation)

---

## Installation

- Clone this repo with

```shell
git clone https://github.com/kenhuang41/cos429-final-project
```

---

## Dataset

The dataset can be found in the folder /cello_pics/ and comprises of 132 images hand-extracted from videos found on Youtube. Credit to the original content can be found in the acknowledgement section of the final report.

---

## Model

The resulting k-nearest neighbors model we trained is in the file "string_predict_knn.sav".

---

## Evaluation

You can qualitatively see the results in the folder /cello_string_results/ with preliminary results in /cello_with_both/, /cello_with_bow/, and /cello_with_fingerboard/. You may need to download the mp4 files to view them.
