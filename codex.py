

from sklearn import svm
import multiprocessing
import pickle
from os.path import join
import re, os

import enum

def register_feature(pattern):
    flags = re.MULTILINE | re.DOTALL
    return re.compile(pattern, flags)


MARKERS = [
    register_feature(r';$'),
    register_feature(r'//[^/]+$'),
    register_feature(r'/\*(?:.*?)\*/'),
    register_feature(r'--'),
    register_feature(r'^module'),
    register_feature(r'require\s+' + '(\'|")'),
    register_feature(r'^import'),
    register_feature(r'#include\s+("|<)'),
    register_feature(r'return\s+\w+;?$'),
    register_feature(r'local\s+\w+\s*='),
    register_feature(r'var\s+\w+\s*='),
    register_feature(r'\$\w+'),
    register_feature(r'std::'),
    register_feature(r'\*\w+|\w+\*'),
    register_feature(r'&&|\|\||~|>>|<<'),
    register_feature(r'\+=|-=|/=|\*=|==|!='),
    register_feature(r'^\s*class'),
    register_feature(r'^\s*public:'),
    register_feature(r'^\s*private:'),
    register_feature(r'^\s*protected:'),
    register_feature(r'this->'),
    register_feature(r'\w+->'),
    register_feature(r'm_\w+(\.|->)'),
    register_feature(r'try\s*{'),
    register_feature(r'}\s*catch'),
    register_feature(r'namespace\s*{'),
    register_feature(r'static_assert\('),
    register_feature(r'static_cast<'),
    register_feature(r'dynamic_cast<'),
    register_feature(r'nullptr'),
    register_feature(r'#define'),
    register_feature(r'#ifdef'),
    register_feature(r'operator::'),
    
    register_feature(r'__\w+'),
    
    # C# markers
    register_feature(r'public\s+[A-Z]+'),
    register_feature(r'protected\s+[A-Z]+'),
    register_feature(r'private\s+[A-Z]+'),
    register_feature(r'internal\s+[A-Z]+'),
    register_feature(r'get\s*{'),
    register_feature(r'set\s*{'),
    register_feature(r'private\s+get\s*{'),
    register_feature(r'private\s+set\s*{'),
    register_feature(r'sealed\s+class'),
    register_feature(r'\s+I[A-Z].\w+'),
    register_feature(r'=>\s+{'),
    register_feature(r'throw new \w+\((.*?)\);\s*$'),

    # JSON markers
    register_feature(r'{\s*"[^"]*?":'),
    register_feature(r':\s*{'),
    register_feature(r':\s*\['),
    register_feature(r'},'),
    register_feature(r'\],'),
    register_feature(r'}}'),
    register_feature(r': "'),

    # XML/HTML markers
    register_feature(r'</?\w+>'),
    register_feature(r'<!--'),

    # HTML markers
    register_feature(r'</?div>'),
    register_feature(r'</?span>'),
    register_feature(r'</?p>'),
    register_feature(r'</?center>'),
    register_feature(r'</!DOCTYPE html>'),
    register_feature(r'<br>'),
    register_feature(r'&nbsp;'),

    # JS markers
    register_feature(r'\s.function\s+\w+\('),
    register_feature(r'\.length'),
    register_feature(r'require\s+\(' + '(\'|")\)'),

    # Haskell markers
    register_feature(r'let\s+\w+\s*='),
    register_feature(r'::\s+\w+\s+->'),
    register_feature(r'>>='),
    
    # Preprocessor markers
    register_feature(r'^\s*#include (<|")'),
    register_feature(r'^\s*#pragma\s'),
    register_feature(r'^\s*#define\s'),
    register_feature(r'^\s*#if(n?def)?\s'),
    register_feature(r'^\s*#undef\s'),
    
    # C/C++ markers
    register_feature(r'\w+\s*\*\s*[a-zA-Z_]\w+'),
    register_feature(r'{$'),
    register_feature(r'^\s*}'),
    register_feature(r'^\s*};'),
    register_feature(r'if\s*\((.*?)\)\s*{'),
    register_feature(r'for\s*\((.*?)\)\s*{'),
    register_feature(r'template\s*<(.*?)>'),
    
    # Delphi markers
    register_feature(r'\(\*(.*?)\*\)'),
    
    #register_feature(r'^\s*@\w+'),
    register_feature(r'\(\)'),
    register_feature(r'\w+::\w+'),
    register_feature(r'^\w*struct\s*(\w+\s*)?{'),
    register_feature(r'\w+:\w+\('),
    
    
    register_feature(r'\\begin'),
    register_feature(r'\\end'),
    register_feature(r'\\\w+\s'),
    register_feature(r'\\\w+({|\[)'),
]


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


def metric(marker, text):
    return sum([m.span()[1] - m.span()[0] for m in re.finditer(marker, text)]) / len(text)


def preprocess(text):
    # TODO: make this unbiased by replacing comment text with the same number of spaces
    text = re.sub('<!--(.*?)-->', '<!-- -->', text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub('"""(.*?)"""', '""" """', text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub('/\*(.*?)\*/', '/* */', text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub('//(.*?)$', '// ', text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub('--(.*?)$', '-- ', text, flags=re.MULTILINE | re.DOTALL)
    #text = re.sub('#(.*?)$', '# ', text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub('"(.*?)[^\\\\]."', '" "', text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub("'(.*?)[^\\\\].'", "' '", text, flags=re.MULTILINE | re.DOTALL)

    text = '\n'.join([x.rstrip() for x in text.split('\n')])

    return text


class Classifier:
    def __init__(self, dataset=None):
        if not dataset:
            self.markers = MARKERS
            self.classif = None
            self.vectors = []
            self.classes = []
        else:
            with open(dataset, 'rb') as datafile:
                self.markers,\
                self.vectors,\
                self.classes,\
                self.classif = pickle.load(datafile)


    def classify(self, text, threshold=0.2):
        if self.classif is None:
            raise ValueError("Classifier not trained.")

        preprocessed_text = preprocess(text)
        weights = [metric(marker, preprocessed_text) for marker in self.markers]
        print("{0:.3f}".format(sum(weights)))  # TODO: debugging
        return Language(self.classif.predict([weights])[0]) if sum(weights) >= threshold else None


    def train(self, texts, languages):
        # TODO: parallelize this later
        for text in texts:
            preprocessed_text = preprocess(text)
            self.vectors.append([metric(m, preprocessed_text) for m in self.markers])

        self.classes += [lang.value for lang in languages]

        self.classif = svm.LinearSVC(class_weight='balanced', multi_class='crammer_singer')
        self.classif.fit(self.vectors, self.classes)  # train the classifier on the dataset


    def save(self, dataset):
        with open(dataset, 'wb') as datafile:
            pickle.dump((self.markers,
                         self.vectors,
                         self.classes,
                         self.classif), datafile, 3)


""" Training helper functions """


def load_dataset(dataset):
    files = {}

    for l in [l for l in os.listdir(dataset) if os.path.isdir(join(dataset, l))]:
        files[l] = [join(dataset, l, f) for f in os.listdir(join(dataset, l))]

    return files


def generate_classifier(dataset, output):
    classifier = Classifier()

    texts = []
    langs = []

    for lang, files in load_dataset(dataset).items():
        for path in files:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = ''.join(f.readlines())
                texts.append(text)
                langs.append(Language(lang))

    classifier.train(texts, langs)
    classifier.save(output)
