# SyllabifyingEnglish

Project Description/Goal: Automatic syllabification refers to the task of adding syllable boundaries to forms. While there are many ways to approach this task, Dinu
et al. 2024 liken it to a sequence labeling task (where graphemes within an inputted word are labeled as either beginning, or not beginning, a syllable), implement a bidirectional gated recurrent units model to perform classification, and achieve an impressive accuracy of 99.74% and F1-score of 99.69 when automatically syllabifying Italian orthographic forms. Italian, however, has a shallow orthography and simple syllable structure. English, by contrast, has a deep orthography and complex syllable structure (Seymour et al., 2003). The goal of the current project, then, was to determine how Dinu et al. 2024’s approach performs on English. 

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

## File Structure
* 25000-syllabified-words-list-master: the raw dataset
* Elman: the Elman model
* GRU: the GRU model
* Analysis: scripts to assess model error
* dataProcessing: script for data preprocessing
* processedData: the training, development, and testing sets
* relevantLiterature: papers related to the project