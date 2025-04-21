# SyllabifyingEnglish

Project Description/Goal: TODO


## Implementation 

**Data**: [25,000 syllabified words list](https://github.com/gautesolheim/25000-syllabified-words-list)


**Elman Model**: A recurrent neural network consisting of ...
* a character embedding layer, producing 96-dimensional vectors for each inputted character
* an Elman RNN cell with an 192-dimension hidden state
* a final linear layer which produces tag scores for each character
* trained using Cross Entropy Loss with the Adam optimzer, a learning rate of 0.001, and 15 training epochs


**GRU Model**: A recurrent neural network consisting of ...
* a character embedding layer, producing 96-dimensional vectors for each inputted character
* a stacked bidirectional GRU with 3 layers, a 96-dimension hidden state, and 0.2-rate dropout between GRU layers
* 0.5-rate dropout applied to the GRU output
* layer normalization applied to the GRU output
* a time-distributed, fully-connected linear layer with ReLU activation, which projects each time step/inputted character onto the tag set 
* a final linear layer which produces tag scores for each character
* trained using Cross Entropy Loss with the Adam optimzer, a learning rate of 0.001, and 15 training epochs

**Grid Search**: Hyper-parameters for both models were found with an identical grid search loop targeting ... 
* embedding dimension = [32, 64, 96]
* hidden dimension (of Elman cell or GRU) = [96, 128, 192]
* learning rate = [0.01, 0.001, 0.0001]
* epochs = [5, 10, 15]
