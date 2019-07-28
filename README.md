Identifying and Categorising Offensive Language in Social Media

The project consisted of following parts:
- Offensive language identification
- Categorization into offense types
- Offense target identification.

Experimented with bi-LSTM attention layer and Hierarchical Attention network (HAN) and combined it into an ensemble model.
Used state-of-the-art word embeddings like ELMO and BERT.

Link to the task: https://competitions.codalab.org/competitions/20011

**Results**

A stratified train-test split of 0.8 was used for all the models:
Results are written in the form: (train, accuracy)

##### Task A

| Model        | Train Accuracy           |
| ------------- |:-------------:|
| BERT      | 0.75 | 
| ConceptNet      | 0.80      |   
| Fasttext | 0.86      |
| Elmo | 0.80 | 
| Glove | 0.79|


##### Task B

| Model        | Train Accuracy           |
| ------------- |:-------------:|
| BERT      | 0.83 | 
| ConceptNet      | 0.75      |   
| Fasttext | 0.96  |
| Elmo | 0.90 | 
| Glove | 0.81|

##### Task C

| Model        | Train Accuracy           |
| ------------- |:-------------:|
| BERT      | 0.75 | 
| ConceptNet      | 0.73      |   
| Fasttext | 0.75      |
| Elmo | 0.70 | 
| Glove | 0.74|


**Project details**

Embeddings used:
- Elmo
- Bert
- Fasttext
- Glove
- ConceptNet

Model architechures used are:
- BiLSTM
- Attention
- Heirchial-Attention-Network

**Future work**
- BERT transformer
- XLnet transformer
- Ensemble modeling