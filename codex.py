from sklearn import svm
import multiprocessing
import re, os, pickle
import numpy as np
import enum, gzip
import os.path


def compiled_regex(pattern):
    flags = re.MULTILINE | re.DOTALL
    return re.compile(pattern, flags)


MARKERS = [
    compiled_regex(r';$'),
    compiled_regex(r'//[^/]+$'),
    compiled_regex(r'/\*(?:.*?)\*/'),
    compiled_regex(r'--'),
    compiled_regex(r'^module'),
    compiled_regex(r'require\s+' + '(\'|")'),
    compiled_regex(r'^import'),
    compiled_regex(r'#include\s+("|<)'),
    compiled_regex(r'return\s+\w+;?$'),
    compiled_regex(r'local\s+\w+\s*='),
    compiled_regex(r'var\s+\w+\s*='),
    compiled_regex(r'\$\w+'),
    compiled_regex(r'std::'),
    compiled_regex(r'\*\w+|\w+\*'),
    compiled_regex(r'&&|\|\||>>|<<'),
    compiled_regex(r'\+=|-=|/=|\*=|==|!='),
    compiled_regex(r'^\s*class'),
    compiled_regex(r'^\s*public:'),
    compiled_regex(r'^\s*private:'),
    compiled_regex(r'^\s*protected:'),
    compiled_regex(r'this->'),
    compiled_regex(r'\w+->'),
    compiled_regex(r'm_\w+(\.|->)'),
    compiled_regex(r'try\s*{'),
    compiled_regex(r'}\s*catch'),
    compiled_regex(r'namespace\s*{'),
    compiled_regex(r'static_assert\('),
    compiled_regex(r'static_cast<'),
    compiled_regex(r'dynamic_cast<'),
    compiled_regex(r'nullptr'),
    compiled_regex(r'#define'),
    compiled_regex(r'#ifdef'),
    compiled_regex(r'operator::'),
    
    compiled_regex(r'__\w+'),
    
    # C# markers
    compiled_regex(r'public\s+[A-Z]+'),
    compiled_regex(r'protected\s+[A-Z]+'),
    compiled_regex(r'private\s+[A-Z]+'),
    compiled_regex(r'internal\s+[A-Z]+'),
    compiled_regex(r'get\s*{'),
    compiled_regex(r'set\s*{'),
    compiled_regex(r'private\s+get\s*{'),
    compiled_regex(r'private\s+set\s*{'),
    compiled_regex(r'sealed\s+class'),
    compiled_regex(r'\s+I[A-Z].\w+'),
    compiled_regex(r'=>\s+{'),
    compiled_regex(r'throw new \w+\((.*?)\);\s*$'),

    # JSON markers
    compiled_regex(r'{\s*"[^"]*?":'),
    compiled_regex(r':\s*{'),
    compiled_regex(r':\s*\['),
    compiled_regex(r'},'),
    compiled_regex(r'\],'),
    compiled_regex(r'}}'),
    compiled_regex(r': "'),

    # XML/HTML markers
    compiled_regex(r'</?\w+>'),
    compiled_regex(r'<!--'),

    # HTML markers
    compiled_regex(r'</?div>'),
    compiled_regex(r'</?span>'),
    compiled_regex(r'</?p>'),
    compiled_regex(r'</?center>'),
    compiled_regex(r'</!DOCTYPE html>'),
    compiled_regex(r'<br>'),
    compiled_regex(r'&nbsp;'),

    # JS markers
    compiled_regex(r'\s.function\s+\w+\('),
    compiled_regex(r'\.length'),
    compiled_regex(r'require\s+\(' + '(\'|")\)'),

    # Haskell markers
    compiled_regex(r'let\s+\w+\s*='),
    compiled_regex(r'::\s+\w+\s+->'),
    compiled_regex(r'>>='),
    
    # Preprocessor markers
    compiled_regex(r'^\s*#include (<|")'),
    compiled_regex(r'^\s*#pragma\s'),
    compiled_regex(r'^\s*#define\s'),
    compiled_regex(r'^\s*#if(n?def)?\s'),
    compiled_regex(r'^\s*#undef\s'),
    
    # C/C++ markers
    compiled_regex(r'\w+\s*\*\s*[a-zA-Z_]\w+'),
    compiled_regex(r'{$'),
    compiled_regex(r'^\s*}'),
    compiled_regex(r'^\s*};'),
    compiled_regex(r'if\s*\((.*?)\)\s*{'),
    compiled_regex(r'for\s*\((.*?)\)\s*{'),
    compiled_regex(r'template\s*<(.*?)>'),
    
    # Delphi markers
    compiled_regex(r'\(\*(.*?)\*\)'),
    
    #compiled_regex(r'^\s*@\w+'),
    compiled_regex(r'\(\)'),
    compiled_regex(r'\w+::\w+'),
    compiled_regex(r'^\w*struct\s*(\w+\s*)?{'),
    compiled_regex(r'\w+:\w+\('),
    
    
    compiled_regex(r'\\begin'),
    compiled_regex(r'\\end'),
    compiled_regex(r'\\\w+\s'),
    compiled_regex(r'\\\w+({|\[)'),
]


PREPROCESSING = {
    compiled_regex(r'<!--(.*?)-->'):    ('<!--',    '-->'),
    compiled_regex(r'"""(.*?)"""'):     ('"""',     '"""'),
    compiled_regex(r"'''(.*?)'''"):     ("'''",     "'''"),
    compiled_regex(r'/\*(.*?)\*/'):     ('/*',      '*/'),
    compiled_regex(r'//(.*?)$'):        ('//',      ''),
    compiled_regex(r'--(.*?)$'):        ('--',      ''),
    compiled_regex(r'"(.*?)"'):         ('"',       '"'),
    compiled_regex(r"'(.*?)'"):         ("'",       "'"),
    compiled_regex(r'#(.*?)$'):         ('#',       ''),
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
    Tex             = 'Tex'
    Json            = 'JSON'


class Classifier:
    def __init__(self, dataset=None):
        if not dataset:
            self.vectors = np.empty((0, len(MARKERS)), float)
            self.classes = np.empty((0,), str)
            self.markers = MARKERS
            self.classif = None
        else:
            with open(dataset, 'rb') as datafile:
                self.markers,\
                self.vectors,\
                self.classes,\
                self.classif = pickle.loads(gzip.decompress(datafile.read()))


    def weight(self, marker, text):
        return sum([m.span()[1] - m.span()[0] for m in re.finditer(marker, text)]) / len(text)


    def measure_weights(self, text):
        text = '\n'.join([line.rstrip() for line in text.split('\n')])

        for pattern, (pre, suf) in PREPROCESSING.items():
            text = re.sub(pattern, lambda m: pre + ' ' * (m.end(1) - m.start(1)) + suf, text)

        return [self.weight(marker, text) for marker in self.markers]


    def classify(self, text, threshold=0.2):
        if self.classif is None:
            raise ValueError("Classifier not trained.")

        weights = self.measure_weights(text)

        if sum(weights) >= threshold:
            return Language(self.classif.predict([weights])[0])
        else:
            return None


    def __call__(self, element):
        return (self.measure_weights(element[0]), element[1].value)


    def train(self, elements):
        with multiprocessing.Pool(processes=None) as par:
            for weights, language in par.map(self, elements):
                self.vectors = np.append(self.vectors, [weights], axis=0)
                self.classes = np.append(self.classes, [language], axis=0)

        self.classif = svm.LinearSVC(class_weight='balanced', multi_class='crammer_singer')
        self.classif.fit(self.vectors, self.classes)  # train the classifier on the dataset


    def save(self, dataset):
        with open(dataset, 'wb') as datafile:
            datafile.write(gzip.compress(pickle.dumps((
                self.markers, self.vectors,\
                self.classes, self.classif))))




""" Training helper functions """




def load_dataset(dataset):
    files = {}

    for l in [l for l in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, l))]:
        files[l] = [os.path.join(dataset, l, f) for f in os.listdir(os.path.join(dataset, l))]

    return files


def generate_classifier(dataset, output):
    classifier = Classifier()

    elements = []

    for lang, files in load_dataset(dataset).items():
        for path in files:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = ''.join(f.readlines())
                elements.append((text, Language(lang)))

    classifier.train(elements)
    classifier.save(output)
