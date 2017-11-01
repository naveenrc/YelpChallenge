### Yelp Dataset challenge
This project is about Natural Language processing and classifying images obtained from yelp dataset which can be downloaded from here https://www.yelp.com/dataset.<br>

### Requirements
NLP
* python 3.5
* spacy
* gensim
* pyLDAvis
* Word2vec
* Bokeh
* tSNE

Image classification
* python 3.5
* tensorflow
* opencv
* 12 GB RAM

## Setup
### NLP
Modern_NLP.ipynb walks through the following topics(<a href="http://nbviewer.jupyter.org/github/naveenrc/YelpChallenge/blob/07f275a4d1a1841dc1141a475fa168edd225b800/Modern_NLP.ipynb">best viewed on nb viewer</a>)
1. A tour of the dataset
2. Introduction to text processing with spaCy
3. Automatic phrase modeling
4. Topic modeling with LDA
5. Visualizing topic models with pyLDAvis
6. Word vector models with word2vec
7. Visualizing word2vec with t-SNE


### Setup for image classification
1. Install project requirements
2. Create a folder yelpData and move the extracted data from yelp into this folder. Move the photos to 'yelpData/yelpPhotos' directory
3. Run photo_process.py, enter the size you desire to resize to Ex: 64 for 64 x 64 or 32 for 32 x 32.
4. Run photo_info.py to get information about the photos
5. Run classifier.py to start the model (may take longer, 6 to 10 hours without a GPU)
6. Run predict.py to predict image label
```bash
pip install -r requirements.txt
python ./photoAnalysis/photo_process.py
python ./photoAnalysis/photo_info.py
python ./classifier/classifier.py
python ./classifier/predict.py
```

<a href="https://drive.google.com/file/d/0BypHvhe9eW_KcWFDa3pyZlNJams/view?usp=sharing">Report for image classification can be found here</a>
