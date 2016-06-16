Codex - Source code language classification
===========================================

This small library is an attempt to classify source code using a support vector machine (SVM) classifier.

It doesn't work very well currently and is missing many languages but is still in development. The current data sets used to train and test the classifier are atrociously bad (non-reviewed random selection of repositories from Github) so the classifier will probably be somewhat off until this is fixed.

Usage
-----

For now please use the pre-trained dataset in `dataset/dataset.bin` (this will be improved later).


To initialize the classifier:

```python
>>> import codex
>>> classifier = codex.Classifier('dataset/dataset.bin')
```

To classify a piece of code:

```python
>>> classifier.classify("""
... #include <stdio.h>
... 
... int main(void) {
...   printf("hello\n");
... }
... """)
<Language.C: 'C'>
```

The `classify` method will return `None` or a best guess if it cannot ascertain the language. In particular, it will return `None` if the input isn't code but is ordinary text.

The classifier can be retrained on additional data with `cls.train`.

Supported languages
-------------------

The accuracy table below is calculated by running the classifier trained with the default dataset on a representative test set, and the percentages are obtained as `(# <language> files detected as <language>) / (# <language> files) * 100`.

Here "None" can be read as "anything that isn't source code", and should be interpreted that way throughout the library.

| Language           | Accuracy | Closest Language   |
| ------------------ | -------- | ------------------ |
| TeX                |   97.5%  |   2.5% None        |
| Java               |   98.4%  |   0.7% C#          |
| Javascript         |   94.2%  |   3.3% Lua         |
| C                  |   99.4%  |   0.2% XML         |
| HTML               |   91.1%  |   7.2% XML         |
| PHP                |   95.8%  |   2.3% C           |
| None               |   88.2%  |   6.5% TeX         |
| JSON               |   96.9%  |   3.1% None        |
| C#                 |   94.4%  |   3.8% Java        |
| C++                |   82.3%  |  11.8% C           |
| Python             |   97.2%  |   1.1% Ruby        |
| Haskell            |   98.7%  |   0.4% Lua         |
| Make               |   84.6%  |  14.7% None        |
| Ruby               |   93.8%  |   2.9% Lua         |
| XML                |   94.5%  |   2.7% HTML        |
| Lua                |   93.1%  |   2.9% None        |
| Delphi             |   97.4%  |   0.8% None        |

TODO
----

- [ ] add proper support for the most popular languages
- [x] improve script to train classifier from a dataset folder
- [x] write scripts to test the classifier against a test set
- [ ] write scripts to generate a training set (code corpus from github)?
- [ ] better documentation
- [ ] better test set

How it works
------------

The motivation for writing this library was to develop a [Discord](https://discordapp.com/) bot that detects when someone posts code, automatically formats it with backticks and sets the proper language. This requires that the bot be able to:

  1. distinguish code snippets from legitimate conversations (most important)
  2. recognize which language a code snippet is written in (less important)

The most obvious way to find out if text is of such and such programming language is to just attempt to parse it according to the language grammar. However, this approach quickly runs into limitations once one realizes that the input may not be a complete block of code (for instance just a small snippet of a larger source code file) and may not necessarily be valid code according to the rules of the language, especially for languages with many different dialects.

This library takes a different approach; instead of rigorously parsing the input text, it attempts to tokenize it according to a chosen set of patterns, called **markers**, which are likely to occur in different programming languages (regular expressions are used for pattern matching). An example of a marker is the following `re.MULTILINE | re.DOTALL` regex

```
^\s*#\s*include\s+("|<)[^">]+("|>)$
```

which matches C/C++ `#include` statements. The basic algorithm then goes as follows:

1) For each marker, find all non-overlapping matches of the pattern in the input, calculate how many characters of the input were matched, and divide this number by the length of the text in characters; this is the *weight* of the text for that marker.

2) Sum up all the marker weights to obtain the text's *score*; if this score - which in some vague sense describes how much (of) the input text looks like code - is below some configurable threshold, then terminate by rejecting the input text as not code.

3) Otherwise, determine the closest language using the weights with a linear support vector machine (SVM) classifier.

Some preprocessing takes place on the input text prior to the algorithm above. In particular, free-form syntactic elements such as comments are removed and replaced with their delimiters, for instance everything enclosed in `/*` and `*/` is replaced with `/* */`; these can then be used later on to help identify the language. String literals are left alone as they often show up in normal text and are unreliable to remove at best.

There is a special case where such constructs can lead to severe ambiguity between languages. An example is the Delphi curly brace comment construct, which represents a compound statement in the C family of languages. These are removed **only** if the matched text **doesn't** look like code according to steps (1) and (2) above. Fortunately most cases are not ambiguous, the only known ambiguous cases are given below:

  * `{...}`: Block comment in Delphi, compound statement in C-like languages

As a general rule, misclassifying the input as the wrong language is preferable to mistakenly detecting normal text as code and vice versa, whenever possible (which is why the threshold value needs to be carefully tuned according to your dataset).
