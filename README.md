# Machine Translation from Kannada to Telugu
Cloud Computing and Big Data project on Machine Translation from one regional language (Kannada) to another (Telugu). 

 Machine Translation pertains to translation of one natural language to other by using automated computing. MT tools are often used to translate vast amounts of information involving millions of words that could not possibly be translated the traditional way.

In human languages, the meaning of a sentence is constructed by composing small chunks of words together with each other, obtaining successively larger chunks with more complex meanings until the sentence is formed in its entirety. The order in which these chunks are combined creates a tree-structured hierarchy corresponding to the sentence. We refer to each sentence’s tree-structured hierarchy as a parse tree, and the phenomenon broadly as syntax.

In recent years, however, neural networks used in NLP have represented each word in the sentence as a realvalued vector, with no explicit representation of the parse tree. By finding the right linear transformation of the points, it can be found that the tree constructed by connecting each word to the word closest to it approximates the human parse tree that can be drawn.


## Methodology

The first stage of this project was to build the structural probe (or obtain the transformation matrix B) from the BERT (Bidirectional Encoder Representations from Transformers) embeddings of an English sentence. This transformation matrix converts the Euclidean distances between words to syntax tree distances. When the structural probe is applied to any English sentence, it can generate the syntax tree for that sentence. A different transformation matrix can be developed for different languages, thus giving us the ability to generate parse trees for a sentence in any language. 

Using the parse trees, a translation of a particular sentence from one word to another can be found. Using BERT, embeddings for a sentence are obtained. Utilizing these embeddings and pre-trained syntax trees, the transformation to convert the Euclidean distances to syntax or parse tree distances can be applied. In order to arrive at a structural probe for a language, we need to apply gradient descent on a rough approximation of the matrix to minimize the error between the tree distances and transformed distances.



## Code

[temporary_final.py](https://github.com/GreeshmaKaranth/BigData/tree/master/Code) contains the code that tries to find a structural probe for the English language using gradient descent.

## Report

A detailed [report](https://github.com/GreeshmaKaranth/BigData/blob/master/Report/MachineTranslationFromKannadaToTelugu.pdf) of the project.

## References

Some papers related to Machine Translations. This project is based off of [StructuralProbe.pdf](https://github.com/GreeshmaKaranth/BigData/blob/master/References/StructuralProbe.pdf)
