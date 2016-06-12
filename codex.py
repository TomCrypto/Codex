from warnings import warn
from sklearn import svm
import multiprocessing
import re, os, pickle
import numpy as np
import enum, gzip
import os.path


def compiled_regex(pattern, dotall=True):
    flags = (re.MULTILINE | re.DOTALL) if dotall else re.MULTILINE
    return re.compile(pattern, flags)


MARKERS = [
    # Markers applicable to several languages
    
    compiled_regex(r'^\s{2,}\S'), # indentation
    compiled_regex(r'.{,2}\s*[=/\*-\+&\|%@<>{}\[\](\)~`_\^;#]+\s*.{,2}'),  # generic symbol capture

    # C preprocessor markers
    
    compiled_regex(r'^\s*#\s*include\s+("|<)[^">]+("|>)$'),
    compiled_regex(r'^\s*#\s*ifn?def\s+\w+$'),
    compiled_regex(r'^\s*#\s*if\s+(.*?)$'),
    compiled_regex(r'^\s*#\s*if\s+defined\((.*?)$'),
    compiled_regex(r'^\s*#\s*define \w+(.*?)$'),
    compiled_regex(r'^\s*#\s*endif$'),
    compiled_regex(r'^\s*#\s*undef\s+\w+$'),
    compiled_regex(r'^\s*#\s*else$'),
    compiled_regex(r'^\s*#\s*pragma(.*?)$'),
    
    # Delphi markers
    
    # TODO: Delphi preprocessor markers
    compiled_regex(r'^unit\s+\w;$'),
    compiled_regex(r'^interface(\s+^uses(.*?))?;$'),
    compiled_regex(r'\w+\s*=\s*(.*?);'),
    compiled_regex(r'^\s*\w+\s*=\s*class\(\w+\)$'),
    compiled_regex(r'^\s*\w+\s*=\s*class\(\w+\)$(.*?)^\s*end;$'),
    compiled_regex(r'\s*\w+:\s*(Integer|integer|String|string|Boolean|boolean|Byte|byte|ShortInt|shortint|Word|word|SmallInt|smallint|LongWord|longword|Cardinal|cardinal|LongInt|longint|Int64|int64|Single|single|Double|double|Currency|currency|Extended|extended|Char|char|WideChar|widechar|AnsiChar|ansichar|ShortString|shortstring|AnsiString|ansistring|WideString|widestring|T\w+)(;|\))'),
    compiled_regex(r'(override|virtual|Override|Virtual|Overload|overload|Cdecl|cdecl|Stdcall|stdcall);'),
    compiled_regex(r'^\s*function\s*\w+(\((.*?)\))?\s*:\s*\w+;'),
    compiled_regex(r'^\s*procedure\s*\w+(\((.*?)\))?;'),
    compiled_regex(r'^\s*property\s+\w+\s*:\s*\w+(.*?);'),
    compiled_regex(r'^\s*constructor Create;'),
    compiled_regex(r'^\s*destructor Destroy;'),
    compiled_regex(r'^\s*var(.*?)^\s*begin'),
    compiled_regex(r'inherited(\s+\w+(\((.*?)\))?)?;'),
    compiled_regex(r'^\s*begin(.*?)^\s*end'),
    compiled_regex(r'\w+\s*:=\s*(.*?);'),
    compiled_regex(r'<>'),
    
    # Python markers
    
    compiled_regex(r'^(\s*from\s+[\.\w]+)?\s*import\s+[\*\.,\w]+(,\s*[\*\.,\w]+)*(\s+as\s+\w+)?$'),
    compiled_regex(r'^\s*def\s+\w+\((.*?):$'),
    compiled_regex(r'^\s*if\s(.*?):$(.*?)(^\s*else:)?$'),
    compiled_regex(r'^\s*if\s(.*?):$(.*?)(^\s*elif:)?$'),
    compiled_regex(r'^\s*try:$(.*?)^\s*except(.*?):'),
    compiled_regex(r'True|False'),
    compiled_regex(r'==\s+(True|False)'),
    compiled_regex(r'is\s+(None|True|False)'),
    compiled_regex(r'if\s+(\S*?)\s+in'),
    compiled_regex(r'^\s*return$'),
    compiled_regex(r'^\s*return\s+\w+(,\s+\w+)*$'),
    compiled_regex(r'^\s*pass$'),
    compiled_regex(r'print\((.*?)\)$'),
    compiled_regex(r'^\s*for\s+\w+\s+in\s+(.*?):$'),
    compiled_regex(r'^\s*class\s+\w+\s*(\([.\w]+\))?:$'),
    compiled_regex(r'^\s*@(staticmethod|classmethod|property)$'),
    compiled_regex(r'__repr__'),
    compiled_regex(r'"(.*?)"\s+%\s+(.*?)$', dotall=False),
    compiled_regex(r"'(.*?)'\s+%\s+(.*?)$", dotall=False),
    compiled_regex(r'^\s*raise\s+\w+Error(.*?)$'),

    # TODO: everything below needs to be reviewed

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

    # ???

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


SPECIAL = {
    compiled_regex(r'<!--(.*?)-->'):    False,
    compiled_regex(r'"""(.*?)"""'):     False,
    compiled_regex(r"'''(.*?)'''"):     False,
    compiled_regex(r'/\*(.*?)\*/'):     False,
    compiled_regex(r'//(.*?)$'):        False,
    compiled_regex(r'-- (.*?)$'):       False,
    compiled_regex(r'--\[\[(.*?)\]\]'): False,
    compiled_regex(r'"(.*?)"'):         False,
    compiled_regex(r"'(.*?)'"):         False,
    compiled_regex(r'\s# (.*?)$'):      False,
    compiled_regex(r'\(\*(.*?)\*\)'):   False,
    compiled_regex(r'{(.*?)}'):         True,
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
    MIN_CHARACTERS = 50
    DEFAULT_THRESHOLD = 0.3

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
                warn("Loaded dataset will override threshold parameter.", RuntimeWarning)


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




""" Training helper functions """

# TODO: improve the below





def load_dataset(dataset):
    files = {}

    for l in [l for l in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, l))]:
        files[l] = [os.path.join(dataset, l, f) for f in os.listdir(os.path.join(dataset, l))]

    return files


def generate_classifier(dataset, output):
    classifier = Classifier(threshold=0.5)

    elements = []

    for lang, files in load_dataset(dataset).items():
        for path in files:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = ''.join(f.readlines())
                elements.append((text, Language(lang)))

    classifier.train(elements)
    classifier.save(output)
