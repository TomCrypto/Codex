from sklearn import svm
import multiprocessing
import numpy as np
import re, pickle
import enum, gzip
import warnings


def _compiled_regex(pattern, dotall=True):
    flags = (re.MULTILINE | re.DOTALL) if dotall else re.MULTILINE
    return re.compile(pattern, flags)


MARKERS = [
    # Markers applicable to several languages

    _compiled_regex(r'^\s{2,}\S'), # indentation

    # TODO: need to explain what is going on here as it's not obvious
    _compiled_regex(r'.{,2}\s*[=/\*-\+\|<\{\}\[\](\)~`_\^;#]+\s*.{,2}'),  # generic symbol capture

    # C preprocessor markers

    _compiled_regex(r'^\s*#\s*include\s+("|<)[^">]+("|>)$'),
    _compiled_regex(r'^\s*#\s*ifn?def\s+\w+$'),
    _compiled_regex(r'^\s*#\s*if\s+(.*?)$'),
    _compiled_regex(r'^\s*#\s*if\s+defined\((.*?)$'),
    _compiled_regex(r'^\s*#\s*define \w+(.*?)$'),
    _compiled_regex(r'^\s*#\s*endif$'),
    _compiled_regex(r'^\s*#\s*undef\s+\w+$'),
    _compiled_regex(r'^\s*#\s*else$'),
    _compiled_regex(r'^\s*#\s*pragma(.*?)$'),

    # Delphi markers

    # TODO: Delphi preprocessor markers
    _compiled_regex(r'{\$(.*?)}$'),
    _compiled_regex(r'^unit\s+\w;$'),
    _compiled_regex(r'^interface(\s+^uses(.*?))?;$'),
    _compiled_regex(r'\w+\s*=\s*(.*?);'),
    _compiled_regex(r'^\s*\w+\s*=\s*class\(\w+\)$'),
    _compiled_regex(r'^\s*\w+\s*=\s*class\(\w+\)$(.*?)^\s*end;$'),
    _compiled_regex(r'\s*\w+:\s*(Integer|integer|String|string|Boolean|boolean|Byte|byte|ShortInt|shortint|Word|word|SmallInt|smallint|LongWord|longword|Cardinal|cardinal|LongInt|longint|Int64|int64|Single|single|Double|double|Currency|currency|Extended|extended|Char|char|WideChar|widechar|AnsiChar|ansichar|ShortString|shortstring|AnsiString|ansistring|WideString|widestring|T\w+)(;|\))'),
    _compiled_regex(r'(override|virtual|Override|Virtual|Overload|overload|Cdecl|cdecl|Stdcall|stdcall);'),
    _compiled_regex(r'^\s*function\s*\w+(\((.*?)\))?\s*:\s*\w+;'),
    _compiled_regex(r'^\s*procedure\s*\w+(\((.*?)\))?;'),
    _compiled_regex(r'^\s*property\s+\w+\s*:\s*\w+(.*?);'),
    _compiled_regex(r'^\s*constructor Create;'),
    _compiled_regex(r'^\s*destructor Destroy;'),
    _compiled_regex(r'^\s*var(.*?)^\s*begin'),
    _compiled_regex(r'inherited(\s+\w+(\((.*?)\))?)?;'),
    _compiled_regex(r'^\s*begin(.*?)^\s*end'),
    _compiled_regex(r'\w+\s*:=\s*(.*?);'),
    _compiled_regex(r'<>'),

    # Python markers

    _compiled_regex(r'^(\s*from\s+[\.\w]+)?\s*import\s+[\*\.,\w]+(,\s*[\*\.,\w]+)*(\s+as\s+\w+)?$'),
    _compiled_regex(r'^\s*def\s+\w+\((.*?):$'),
    _compiled_regex(r'^\s*if\s(.*?):$(.*?)(^\s*else:)?$'),
    _compiled_regex(r'^\s*if\s(.*?):$(.*?)(^\s*elif:)?$'),
    _compiled_regex(r'^\s*try:$(.*?)^\s*except(.*?):'),
    _compiled_regex(r'True|False'),
    _compiled_regex(r'==\s+(True|False)'),
    _compiled_regex(r'is\s+(None|True|False)'),
    _compiled_regex(r'if\s+(\S*?)\s+in'),
    _compiled_regex(r'^\s*return$'),
    _compiled_regex(r'^\s*return\s+\w+(,\s+\w+)*$'),
    _compiled_regex(r'^\s*pass$'),
    _compiled_regex(r'print\((.*?)\)$'),
    _compiled_regex(r'^\s*for\s+\w+\s+in\s+(.*?):$'),
    _compiled_regex(r'^\s*class\s+\w+\s*(\([.\w]+\))?:$'),
    _compiled_regex(r'^\s*@(staticmethod|classmethod|property)$'),
    _compiled_regex(r'__repr__'),
    _compiled_regex(r'"(.*?)"\s+%\s+(.*?)$', dotall=False),
    _compiled_regex(r"'(.*?)'\s+%\s+(.*?)$", dotall=False),
    _compiled_regex(r'^\s*raise\s+\w+Error(.*?)$'),

    # Haskell markers

    _compiled_regex(r'let\s+\w+\s*='),
    _compiled_regex(r'::\s+\w+\s+->'),
    _compiled_regex(r'>>='),
    _compiled_regex(r'^\s*import(\s+qualified)?\s+[\.\w]+(\s*\((.*?))?$'),
    _compiled_regex(r'^\s*module\s+[\.\w]+(.*?)where$'),
    _compiled_regex(r'^\s*{-#(.*?)#-}'),
    _compiled_regex(r'^\s*\w+\s*::(.*?)$'),
    _compiled_regex(r'->\s*\[?[\w]+\]?'),
    _compiled_regex(r'\w+\s*<-\s*\w+'),
    _compiled_regex(r'\w+\s+\$\s+\w+'),
    _compiled_regex(r'\(\w+::\w+\)'),
    _compiled_regex(r"\w+'"),
    _compiled_regex(r'<\$>'),
    _compiled_regex(r'=>'),
    _compiled_regex(r'^\s*instance\s+\w+\s+\w+=>(.*?)where$'),
    _compiled_regex(r'^(.*?)=\s+do$', dotall=False),
    _compiled_regex(r'\+\+'),
    _compiled_regex(r'where$'),
    _compiled_regex(r'^\s*\|\s+\w+(.*?)=(.*?)$'),

    # XML markers

    _compiled_regex(r'</?\w+>'),
    _compiled_regex(r'<\?xml(.*?)\?>'),

    # HTML markers

    _compiled_regex(r'<script>(.*?)</script>'),
    _compiled_regex(r'<style>(.*?)</style>'),
    _compiled_regex(r'<link>(.*?)</link>'),
    _compiled_regex(r'<title>(.*?)</title>'),
    _compiled_regex(r'<center>(.*?)</center>'),
    _compiled_regex(r'</!DOCTYPE html(.*?)>'),
    _compiled_regex(r'<br>'),
    _compiled_regex(r'&nbsp;'),
    _compiled_regex(r'<div(\s+[\.\-\w]+="[^"]*")*>(.*?)</div>'),
    _compiled_regex(r'<span(\s+[\.\-\w]+="[^"]*")*>(.*?)</span>'),
    _compiled_regex(r'<p(\s+[\.\-\w]+="[^"]*")*>(.*?)</p>'),
    _compiled_regex(r'<ul(\s+[\.\-\w]+="[^"]*")*>(.*?)</ul>'),
    _compiled_regex(r'<ol(\s+[\.\-\w]+="[^"]*")*>(.*?)</ol>'),
    _compiled_regex(r'<li(\s+[\.\-\w]+="[^"]*")*>(.*?)</li>'),
    _compiled_regex(r'<pre(\s+[\.\-\w]+="[^"]*")*>(.*?)</pre>'),
    _compiled_regex(r'<h\d(\s+[\.\-\w]+="[^"]*")*>(.*?)</h\d>'),
    _compiled_regex(r'<table(\s+[\.\-\w]+="[^"]*")*>(.*?)</table>'),
    _compiled_regex(r'<tr(\s+[\.\-\w]+="[^"]*")*>(.*?)</tr>'),
    _compiled_regex(r'<td(\s+[\.\-\w]+="[^"]*")*>(.*?)</td>'),
    _compiled_regex(r'<img(.*?)>'),

    # JSON markers

    _compiled_regex(r'(,|{|\[)?\s*"[^"]*"\s*:\s*\['),
    _compiled_regex(r'(,|{|\[)?\s*"[^"]*"\s*:\s*\[]'),
    _compiled_regex(r'(,|{|\[)?\s*"[^"]*"\s*:\s*{'),
    _compiled_regex(r'(,|{|\[)?\s*"[^"]*"\s*:\s*{}'),
    _compiled_regex(r'(,|{|\[)?\s*"[^"]*"\s*:\s*[\.\-\deE]+\s*(,|}|\])'),
    _compiled_regex(r'(,|{|\[)?\s*"[^"]*"\s*:\s*"[^"]*"\s*(,|}|\])'),
    _compiled_regex(r'(,|{|\[)?\s*"[^"]*"\s*:\s*true\s*(,|}|\])'),
    _compiled_regex(r'(,|{|\[)?\s*"[^"]*"\s*:\s*false\s*(,|}|\])'),
    _compiled_regex(r'(,|{|\[)?\s*"[^"]*"\s*:\s*null\s*(,|}|\])'),
    _compiled_regex(r'(({|\[)\s({|\[))+'),
    _compiled_regex(r'((}|\])\s(}|\]))+'),

    # Javascript markers

    _compiled_regex(r'\w+\.get(.*?);'),
    _compiled_regex(r'\w+:\s*function\s*\((.*?)},?'),
    _compiled_regex(r'this\.\w+'),
    _compiled_regex(r'var\s+\w+(\s*,\s*\w+)*\s*=(.*?);$'),
    _compiled_regex(r'[\.\w+]+\s*===\s*[\.\w+]+'),
    _compiled_regex(r'require\s*\((.*?)\);?'),
    _compiled_regex(r'undefined'),
    _compiled_regex(r'\.length'),
    _compiled_regex(r'\$\((.*?)\);'),





    # TODO: everything below needs to be reviewed

    _compiled_regex(r'^module'),
    _compiled_regex(r'require\s+' + '(\'|")'),
    _compiled_regex(r'^import'),
    _compiled_regex(r'#include\s+("|<)'),
    _compiled_regex(r'return\s+\w+;?$'),
    _compiled_regex(r'local\s+\w+\s*='),
    _compiled_regex(r'var\s+\w+\s*='),
    _compiled_regex(r'\$\w+'),
    _compiled_regex(r'std::'),
    #_compiled_regex(r'\*\w+|\w+\*'),
    _compiled_regex(r'&&|\|\||>>|<<'),
    _compiled_regex(r'\+=|-=|/=|\*=|==|!='),
    _compiled_regex(r'^\s*class'),
    _compiled_regex(r'^\s*public:'),
    _compiled_regex(r'^\s*private:'),
    _compiled_regex(r'^\s*protected:'),
    _compiled_regex(r'this->'),
    _compiled_regex(r'\w+->'),
    _compiled_regex(r'm_\w+(\.|->)'),
    _compiled_regex(r'try\s*{'),
    _compiled_regex(r'}\s*catch'),
    _compiled_regex(r'namespace\s*{'),
    _compiled_regex(r'static_assert\('),
    _compiled_regex(r'static_cast<'),
    _compiled_regex(r'dynamic_cast<'),
    _compiled_regex(r'nullptr'),
    _compiled_regex(r'operator::'),

    _compiled_regex(r'__\w+'),

    # C# markers
    _compiled_regex(r'public\s+[A-Z]+'),
    _compiled_regex(r'protected\s+[A-Z]+'),
    _compiled_regex(r'private\s+[A-Z]+'),
    _compiled_regex(r'internal\s+[A-Z]+'),
    _compiled_regex(r'get\s*{'),
    _compiled_regex(r'set\s*{'),
    _compiled_regex(r'private\s+get\s*{'),
    _compiled_regex(r'private\s+set\s*{'),
    _compiled_regex(r'sealed\s+class'),
    _compiled_regex(r'\s+I[A-Z].\w+'),
    _compiled_regex(r'=>\s+{'),
    _compiled_regex(r'throw new \w+\((.*?)\);\s*$'),

    # C/C++ markers
    _compiled_regex(r'\w+\s*\*\s*[a-zA-Z_]\w+'),
    _compiled_regex(r'{$'),
    _compiled_regex(r'^\s*}'),
    _compiled_regex(r'^\s*};'),
    _compiled_regex(r'if\s*\((.*?)\)\s*{'),
    _compiled_regex(r'for\s*\((.*?)\)\s*{'),
    _compiled_regex(r'template\s*<(.*?)>'),

    # ???

    #_compiled_regex(r'^\s*@\w+'),
    _compiled_regex(r'\(\)'),
    _compiled_regex(r'\w+::\w+'),
    _compiled_regex(r'^\w*struct\s*(\w+\s*)?{'),
    _compiled_regex(r'\w+:\w+\('),

    # TeX markers

    _compiled_regex(r'\\begin'),
    _compiled_regex(r'\\end'),
    _compiled_regex(r'\\\w+\s'),
    _compiled_regex(r'\\\w+({|\[)'),
]


SPECIAL = {
    _compiled_regex(r'<!--(.*?)-->'):    False,
    _compiled_regex(r'"""(.*?)"""'):     False,
    _compiled_regex(r"'''(.*?)'''"):     False,
    _compiled_regex(r'/\*(.*?)\*/'):     False,
    _compiled_regex(r'//(.*?)$'):        False,
    _compiled_regex(r'-- (.*?)$'):       False,
    _compiled_regex(r'--\[\[(.*?)\]\]'): False,
    _compiled_regex(r'"(.*?)"'):         False,
    _compiled_regex(r"'(.*?)'"):         False,
    _compiled_regex(r'\s# (.*?)$'):      False,
    _compiled_regex(r'\(\*(.*?)\*\)'):   False,
    _compiled_regex(r'{(.*?)}'):         True,
}


@enum.unique  # no dupes
class Language(enum.Enum):
    C               = 'C'
    Haskell         = 'Haskell'
    Cpp             = 'C++'
    Python          = 'Python'
    Java            = 'Java'
    Lua             = 'Lua'
    CSharp          = 'C#'
    Ruby            = 'Ruby'
    Php             = 'PHP'
    Delphi          = 'Delphi'
    Javascript      = 'Javascript'
    Xml             = 'XML'
    Html            = 'HTML'
    Tex             = 'TeX'
    Json            = 'JSON'


class Classifier:
    MIN_CHARACTERS = 40
    DEFAULT_THRESHOLD = 0.25

    def __init__(self, dataset=None, threshold=None):
        if not threshold:
            threshold = Classifier.DEFAULT_THRESHOLD

        if not dataset:
            self.vectors = np.empty((0, len(MARKERS)), float)
            self.classes = np.empty((0,), str)
            self.markers = MARKERS
            self.threshold = threshold
            self.classif = None
        else:
            with open(dataset, 'rb') as datafile:
                self.markers,\
                self.vectors,\
                self.classes,\
                self.classif,\
                self.threshold = pickle.loads(gzip.decompress(datafile.read()))

            if self.threshold != threshold:
                warnings.warn("Loaded dataset will override threshold parameter.", RuntimeWarning)

        if (self.threshold <= 0.0) or (self.threshold >= 2.0):
            raise ValueError("Threshold value out of bounds.")


    def weight(self, marker, text):
        return sum([m.end() - m.start() for m in re.finditer(marker, text)]) / len(text)


    def weight_vector(self, text):
        return [self.weight(marker, text) for marker in self.markers]


    def is_comment(self, text):
        if len(text) >= Classifier.MIN_CHARACTERS:
            return sum(self.weight_vector(text)) < self.threshold
        else:
            return True


    def measure_weights(self, text):
        text = '\n'.join([line.rstrip() for line in text.split('\n')])
        text.replace('\\\n', ' ')  # need to escape trailing backslash

        for pattern, ambiguous in SPECIAL.items():
            for match in re.finditer(pattern, text):
                if not ambiguous or self.is_comment(match.group(1)):
                    text = text[:match.start(0)] + ' ' + text[match.end(0):]

        return self.weight_vector(text)


    def classify(self, text):
        if self.classif is None:
            raise ValueError("Classifier not trained.")

        if len(text) < Classifier.MIN_CHARACTERS:
            return None

        weights = self.measure_weights(text)

        if sum(weights) >= self.threshold:
            return Language(self.classif.predict([weights])[0])
        else:
            return None


    def __call__(self, element):
        if len(element[0]) >= Classifier.MIN_CHARACTERS:
            return (self.measure_weights(element[0]), element[1].value)
        else:  # reject the training item if it is too short
            return (None, None)


    def train(self, elements):
        with multiprocessing.Pool(processes=None) as par:
            for weights, language in par.map(self, elements):
                if weights and language:  # check item wasn't rejected
                    self.vectors = np.append(self.vectors, [weights], axis=0)
                    self.classes = np.append(self.classes, [language], axis=0)

        self.classif = svm.LinearSVC(class_weight='balanced', multi_class='crammer_singer')
        self.classif.fit(self.vectors, self.classes)  # train the classifier on the dataset


    def save(self, dataset):
        with open(dataset, 'wb') as datafile:
            datafile.write(gzip.compress(pickle.dumps((
                self.markers, self.vectors,\
                self.classes, self.classif,
                self.threshold))))


###########################
#### CLI scripts below ####
###########################


_train_descr="""
Script to train a classifier on a folder containing the training
set as a collection of source code files. The layout should be:

    /
    ├── C++
    │   ├── foo.cpp
    │   └── bar.cpp
    └── Java
        └── baz.java

The test set, if provided, must also follow the same layout.

Every file in the language folders will be processed, regardless
of extension, and top-level files in the folder will be ignored.
"""

_classify_descr="""
Script to classify a list of files given on the command line, or
standard input. Each file will be classified and the result will
be displayed.
"""


import argparse
import os.path
import os, sys


def _load_dataset(dataset):
    files = {}

    for l in [l for l in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, l))]:
        files[l] = [os.path.join(dataset, l, f) for f in os.listdir(os.path.join(dataset, l))]

    return files


def _train_classifier(train_path, threshold):
    elements = []

    classifier = Classifier(threshold=threshold)

    for lang, files in _load_dataset(train_path).items():
        for path in files:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                elements.append((''.join(f.readlines()), Language(lang)))

    classifier.train(elements)
    return classifier


def _test(classifier, test_path):
    print('+--------------------+----------+--------------------+')
    print('| Language           | Accuracy | Closest Language   |')
    print('| ------------------ | -------- | ------------------ |')

    for folder, files in _load_dataset(test_path).items():
        tally = {}

        for path in files:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                result = classifier.classify(text = ''.join(f.readlines()))

            tally[result] = (tally[result] if result in tally else 0) + 1

        if Language(folder) not in tally:
            tally[Language(folder)] = 0

        for k in tally:
            tally[k] /= len(files)

        accuracy = tally[Language(folder)] * 100
        del tally[Language(folder)]  # forget it

        closest = max(tally, key=tally.get)
        closest_lang = tally[closest] * 100

        print('| {0: <18} '.format(folder), end='')
        print('|  {0:5.1f}%  '.format(accuracy), end='')
        print('| {0:4.1f}% '.format(tally[closest] * 100), end='')
        print('{0: <12} |'.format(closest.value if closest else "None"))

    print('+--------------------+----------+--------------------+')

def cli_train():
    parser = argparse.ArgumentParser(description=_train_descr,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-x', '--threshold', nargs=1, type=float,
                        default=None, metavar='N',
                        help="the classifier threshold value")
    parser.add_argument('-o', '--out', default=None, metavar='DEST',
                        help="output path for trained dataset")
    parser.add_argument('train', nargs=1, metavar='TRAIN',
                        help="path to the training set folder")
    parser.add_argument('test', nargs='?', metavar='TEST',
                        help="path to the testing set folder")

    args = parser.parse_args()

    if os.path.isfile(args.train[0]):
        print("Training set argument is a file, assuming pre-trained"\
              "; ignoring -x/--threshold and -o/--output arguments...")
        classifier, args.out = Classifier(args.train[0]), None
    else:
        print("Training classifier on `{}'...".format(args.train[0]))
        classifier = _train_classifier(args.train[0], args.threshold)

    if args.test:
        _test(classifier, args.test)

    if args.out:
        print("Saving trained dataset to `{}'...".format(args.out))
        classifier.save(args.out)


def cli_classify():
    parser = argparse.ArgumentParser(description=_classify_descr,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('dataset', nargs=1, metavar='DATASET',
                        help="path to pre-trained dataset")
    parser.add_argument('files', nargs='*', metavar='FILE',
                        help="paths to files to classify")

    args = parser.parse_args()

    classifier = Classifier(args.dataset[0])

    if not args.files:
        args.files = ['-']

    for path in args.files:
        if path == '-':
            text = ''.join(sys.stdin.readlines())
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = ''.join(f.readlines())

        print('{0}\t{1}'.format(path, classifier.classify(text)))
