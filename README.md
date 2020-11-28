# Face Recognition using Facenet architecture

This face recognition system designed based on Facenet architecture. Facenet produces a 128D vector for each face where two same face's euclidian distance is small and two different face's euclidian distance is large comparatively. Here we used SVM and logistic regression to classify faces. The dataset we used is
 [5 celebraty dataset](https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset) from kaggle. But problem using this method is we can not identify a person as an unknown. So for the real time recognition system here i measured the euclidian distance to recognize a person.
 If the distance is higher than a threshold value than our system will detect that face as unknown.

### Prerequisites


```
* Tensorflow == 2.0.0
* scikit-learn == 0.22.1
* opencv == 4.2.0
* dlib == 19.20.0
```

### Usage

To recognize your custom face clone the repository and create a directory in the faces folder with the name of the faces. After installing the dependencies some command should be write on the terminal. 
Below the commands are given.

```
python ./prepare_data.py
python ./embadding_map.py
python ./real_time.py
```

And that's all!


## Author

* **Md. Iftekher Hossain Nabil** [Nabil0036](https://github.com/Nabil0036)

## Acknowledgments

* The [model](https://github.com/nyoki-mtl/keras-facenet) here i used is designed by [nyoki-mtl](https://github.com/nyoki-mtl)
* Spacial Thanks to [Jason](https://www.linkedin.com/in/jasonbrownlee/) and [Adrian](https://www.linkedin.com/in/adrian-rosebrock-59b8732a/)
