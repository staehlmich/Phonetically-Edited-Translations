# Phonetically Edited Translations (PETs)

## Step 1: Create phrase table with Moses

For this step, you need to install the the following packages:
* To install the [MosesToolkit](https://github.com/moses-smt/mosesdecoder), follow the installation guide on the [official website](http://www.statmt.org/moses/?n=Development.GetStarted).

Steps:
1. Train Moses to generate a phrase table. Follow the training steps 1-6 or run the bash script `phrase_table.sh`.
2. Detokenize the phrase table. Moses replaces punctuation symbols with special characters such as *\&quot;* for apostrophe. These characters cannot be phonetized with G2P tools. 

Experiment settings:
* Preprocessing: We chose not to remove sentences with 100 or more tokens.
* Training: In our experiments, we generate phrases with up to 5 tokens, but only use 3-grams in later steps.


## Step 2: Filter phrase table

1. Remove phrases which contain non-alphabetical characters. These characters cannot be converted by G2P tools.
2. Keep phrases with a number of tokens <= 3.
3. Only retain phrases with inverted and direct translation probability >= 0.05.
4. Keep only the top n=5 translations.

## Step 3: Phonetize phrase table

For this step, you need to install the the following packages:
* [Epitran](https://github.com/dmort27/epitran) with `lex_lookup`.
* [ipapy](https://github.com/pettarin/ipapy)

Steps: 
1. Convert the source phrases to IPA strings. Phonetized tokens are not separated by whitespace.
2. Remove suprasegmental symbols with the `IPAString` from `ipapy`, since they cannot be featurized with `Wordkit`.

Note: G2P conversion to ARPABET symbols is implemented with [g2p_en](https://github.com/Kyubyong/g2p), but is not supported in PETS.

## Step 4: Find Phonetically Edited Translations (PETs)
For this step, you need to install the the following packages:
* [python-Levenshtein](https://github.com/ztane/python-Levenshtein)
* [wordkit](https://github.com/clips/wordkit)
* [scipy](https://scipy.org/install/)

Steps:
Input: Transcription of source sentence, target sentence and machine translation output.
1. Using the phonetized phrase table from step 2., find phrases in the source sentence, that do not have a a translation in the output sentence. We call these phrases *candidates*.
2. Search phonetically similar phrases to the *candidates* with a modified Levenshtein distance.
3. Calculate cosine similarity between the candidates and the phonetically similar phrases with [Patpho](https://link.springer.com/article/10.3758/BF03195469). 
   * CVTransformer and ONCTransformer are [Patpho](https://link.springer.com/article/10.3758/BF03195469) implementations in `Wordkit`. 
   * Put all the phonetically similar phrases on a CV (consonant vowel) grid.
   * Vectorize the candidates and similar phrases.
   * Calculate the cosine similarity
4. Retrieve the translations of phrases with a similarity greater than paramter `sim` from the phonetic table. We call these ***Phonetically Edited Translations (PETs)***.
5. Using a naive pattern-matching alignment method, check if the PET appears in the target sentence and is aligned to the source candidate.


Parameters:
* `max_dist` = Threshold for max edit operations with Levenshtein distance between source candidate and phrases.
* `costs` =  Costs for edit operations with Levenshtein distance. (deletion, insertions, substitutions). In our experiments, assign higher insertion costs only at the beginning and end of a string. 
* `sim` = Threshold for mininum cosine similarity score between source candidate and phrases.
* `left` = Alignment on CV grid in `Patpho`.
* `n`= Maximum alignment distance between source candidate and PETs.  


Experiment settings: 
* `max_dist` = 0.6
* `costs` = `(1,2,1)`
* `sim` = 0.7
* `left` = `True`
* `n`= `3`  