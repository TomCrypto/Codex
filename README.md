Codex - Source code language classification
===========================================

This small library is an attempt to classify source code using a support vector machine (SVM) classifier.

It doesn't work very well currently but is still in development.

Usage
-----

TODO: dataset stuff

For now use the pre-trained dataset in `dataset/dataset.bin`.


To initialize the classifier:

```python
>>> import codex
>>> cls = codex.Classifier('dataset/dataset.bin')
```

To classify a piece of code:

```python
>>> cls.classify("""
... #include <stdio.h>
... 
... int main(void) {
...   printf("hello\n");
... }
... """)
<Language.C: 'C'>
```

The `classify` method will return `None` if it cannot ascertain the language. In particular, it will return `None` if the input isn't code but is ordinary text.

The classifier can be retrained on additional data with `cls.train`.
