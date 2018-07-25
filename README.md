## Algorithmic Sonnet Generation
Tianlin Duan, Liuyi Zhu, Peter Hase, John Benhart 

Duke Data Science Team

**Introduction.** Our team provides a program to generate Shakespearean sonnets within 20 minutes on a standard laptop CPU. We build off of the progress of the 2016 Poetix winners [1], taking steps to augment their approach. At a high level, we use an LSTM-based recurrent neural network to evaluate the likelihood of lines which are constrained to follow iambic pentameter, rhyme as necessary, and bear topical resemblance to a user input. We employ a beam search to select lines with high likelihood out of the rather sizable space of possible lines. 

**Instructions.** Generated poems can be found in the *output_poems* folder. To generate a poem, first install the required dependencies (given python 3.6x):

1) Tensorflow	
2) gensim	
3) numpy	
4) argparse	
5) nltk

Then navigate to the submission folder via the command line and execute the following:

*python genPoems.py <topic> <seed>*

where topic is the user-supplied topic of the poem and seed (an optional argument) is an integer for the seed. Each topic should be one word. Additionally, we require the words in the topic to exist in the 6 billion token GloVe dictionary [2].

**Methodology.** A precise description of our approach follows. As a general rule, available words are constrained to those which are metrical (according to the stress encodings of [3]) and which are represented in the language model’s training text.

We first select the poem’s rhyme words. We make use of the user-supplied topic here, finding a GloVe representation of the topic and evaluating a distance between this representation and each rhyme pair (per the 6 billion token GloVe model). The topic-representation is the arithmetic mean of the GloVe vectors corresponding to each word in the topic, and the distance metric is the maximum of cosine distances between topic-representation and the GloVe vectors for the two words in the rhyme pair. We transform the distances into a probability distribution over rhyme pairs and randomly sample one pair at a time, dropping all pairs within the rhyme class consumed with each pair selection and re-normalizing the distribution as necessary. The more similar the topic-representation and a rhyme pair, the more likely it is that this pair is selected for the poem. Rhyme pairs are then randomly ordered at the ends of lines.

Line construction proceeds backwards off of the rhyme words, with each line determined independently. We use a probabilistic language model to evaluate conditional probabilities of a next word given the lines-in-progress, starting with the rhyme word. We search through the space of possible lines with a beam search, pruning unlikely branches at each step. Finally, the chosen line is sampled from the top 14 lines according to their re-normalized probabilities. Along the way, a variety of grammatical restrictions are enforced on the search, primarily involving the part of speech progression. For efficiency, an FSA is used to structure the search, with states containing an in-progress sequence, meter information, a probability, and an LSTM state.

The language model is an LSTM-based recurrent neural network with 3 layers and 512 neurons per layer. Its input is a state and a new word, and its output is an updated state and a probability distribution over its known words. To provide some stylistic variety, we train three distinct models, each with its own training corpus. The corpora are selected works of 1) Robert Frost, 2) Walt Whitman, and 3) Suzanne Collins. They have 7000, 14000, and 7500 unique words, respectively. Regarding implementation, we extend Sung Kim’s work [4]. GloVe embeddings provide the actual input to the model, with a random vector used as a representation for unknown tokens. Each model is intentionally overfit to its training text. Using cosine annealing for learning rate decay, training proceeds with the word embeddings untrainable until the training loss stalls; then the word embeddings are trainable until loss plateaus. At generation time, one of the three models is randomly selected for use.

In a post-processing step, we add punctuation. Beyond the standard end-of-quatrain placement of periods, we place commas within the poem using the language model. The language model learns punctuation placement during training; in post-processing, a number of commas is randomly selected, then we evaluate the likelihood of a single comma in all possible locations and (with an independence assumption) place the pre-selected number into the most likely positions. 

In summary, we select rhyme words based on the user input, build each line backward off of the rhyme words, then post-process for punctuation. 

**Limitations and Conclusions.** Our approach is most limited by a deficiency in handling possessive words, occasionally weak grammar, and a lack of a strong inter-line coherence mechanism. While the language model learns the concept of possession, we were unable to reconcile the use of possessive language with our beam search and metrical constraints. We forewent the use of a strict grammar parsing tool to give the language model greater freedom in line selection. This freedom is sometimes misused. Semantic or grammatical inter-line coherence will require more creative use of the language model or an as yet undefined process. Despite the limitations, however, we believe our poems are often evocative of human poetry. 

**References**

[1] https://www.isi.edu/natural-language/mt/generating-topical-poetry.pdf

[2] https://nlp.stanford.edu/projects/glove/

[3] http://www.speech.cs.cmu.edu/cgi-bin/cmudict

[4] https://github.com/hunkim/word-rnn-tensorflow
