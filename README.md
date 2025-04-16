# SyllabifyingEnglish
Attempting to automatically syllabify English grapheme forms with an RNN. 

**Data**: [25,000 syllabified words list](https://github.com/gautesolheim/25000-syllabified-words-list)


**Elman Model**: An RNN with an embedding layer, one hidden layer and one linear output layer. Fully connected. Masking was performed.

Manual grid search was performed targeting: 
* embedding_dims = [32, 64, 96]
* hidden_dims = [96, 128, 192]
* l_rs = [0.01, 0.001, 0.0001]
* epochs = [5,10,15]

Best model paramters: 