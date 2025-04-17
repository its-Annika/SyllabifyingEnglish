# SyllabifyingEnglish
Attempting to automatically syllabify English grapheme forms with an RNN. 

### Data
[25,000 syllabified words list](https://github.com/gautesolheim/25000-syllabified-words-list)


### Elman Model
 An RNN with an embedding layer, one hidden layer and one linear output layer. Fully connected. Masking was performed.

Manual grid search was performed targeting: 
* embedding_dims = [32, 64, 96]
* hidden_dims = [96, 128, 192]
* l_rs = [0.01, 0.001, 0.0001]
* epochs = [5,10,15]

Best model paramters: 
* embedding_dim = 96
* hidden_dim = 192
* l_r = 0.001
* epochs = 15

Model Performance:
|             | Precision | Recall | F1 | Support|
| ----------- | -------| ----------| ----   |   ---   |
| no boundary | 0.9067 |    0.9446 | 0.9253 |   14468 |
| boundary    |0.7430  |  0.6224   | 0.6774 |   3726  |
|average      |        |           | 0.8786 | 18194   |
|macro avg    | 0.8249 | 0.7835    |0.8014  |  18194  |
|micro avg    |  0.8732|0.8786     |0.8745  | 18194   |