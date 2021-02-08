import os
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

import spacy
import spacy.gold
import spacy.lang.es
import spacy.tokens
import biome.text
from biome.text.errors import WrongValueError
from spacy.tokenizer import Tokenizer

spacy.tokens.doc.Doc.set_extension("entity_text", default=None)
spacy.tokens.token.Token.set_extension("ctag", default=None)

# used only in the predictions
TOKENS_TO_REPLACE_FOR_BERT = ['\n', '\n\n', '\n\n\n', ' ', '\uf0a7']
# from the test set:
#     [
#     '\n',
#     '\n\n',
#     '\n\n\n',
#     '\x85',
#     '\x85 ',
#     '\x99',
#     '\n\xa0\n',
#     '\n\n\xa0\n',
#     '\xad',
#     '\n\n\n\n',
#     '\n\n\n\n\n\n',
#     '�',
#     '\n \n',
#     ' \n\n',
#     '\n\n \n',
#     '\n\n \n\n',
#     ' ',
#     '  ',
#     '\n\n ',
#     '\n \n ',
#     '\n\n \n\n ',
#     '  \n\n',
#     '\u2028\n\n',
#     '\u2028      \n\n',
#     '\u2028\u2028\n',
#     '                        \n\n',
#     '\n\n  ',
#     '\n  ',
#     '\n  \n  \n',
#     '\n \n\n',
#     '\n  \n',
#     '\n  \n\n',
#     '\n \n \n',
#     '\n ',
#     '\t',
#     '\t\t',
#     '\t\t\n',
#     '\t\t\t'
# ]


class Span:
    """Represents a span of an annotation. These spans can be found in BRAT annotations.

    Parameters
    ----------
    start
    stop
    label
    text
    file

    Attributes
    ----------
    start
    stop
    label
    text
    file
    parents
    children
    siblings
    """

    def __init__(
        self,
        start: int,
        stop: int,
        label: str = None,
        text: str = None,
        file: str = None,
    ):
        self.start = start
        self.stop = stop
        self.label = label
        self.text = text
        self.file = file

        self.parents: List[Span] = []
        self.children: List[Span] = []
        self.siblings: List[Span] = []

    @property
    def size(self) -> int:
        """Returns the size of the span"""
        return self.stop - self.start

    def is_child_of(self, span: "Span") -> bool:
        """Checks if self is contained in span"""
        return self.start >= span.start and self.stop <= span.stop and not self == span

    def is_parent_of(self, span: "Span") -> bool:
        """Checks if span is contained in self"""
        return span.start >= self.start and span.stop <= self.stop and not self == span

    def is_sibling_of(self, span: "Span") -> bool:
        """Checks if span overlaps with self. Complete containment will return False."""
        return (span.start < self.start < span.stop < self.stop) or (
            self.start < span.start < self.stop < span.stop
        )

    def add_relatives(self, spans: List["Span"]):
        """Add spans to self: either as parents, as children or as siblings

        Parameters
        ----------
        spans
            List of possible relatives of self
        """
        for span in spans:
            if self.is_child_of(span):
                self.parents.append(span)
            elif self.is_parent_of(span):
                self.children.append(span)
            elif self.is_sibling_of(span):
                self.siblings.append(span)

    def to_spacy(self) -> Tuple[int, int, str]:
        """Returns a tuple that can be used by `spacy.gold.bilou_tags_from_offsets()`

        Returns
        -------
        start, stop, label
        """
        return self.start, self.stop, self.label

    @classmethod
    def from_brat(
        cls, line: str, file: str = None, doc: spacy.tokens.doc.Doc = None
    ) -> "Span":
        """Returns a Span from a brat ann line

        Parameters
        ----------
        line
            A line corresponding to an annotation in a brat ann file
        file
            File name of the brat ann file
        doc
            If provided, aligns the span with the tokens in the doc

        Returns
        -------
        span
        """
        nr_span_text = line.split("\t")
        label, start, stop = nr_span_text[1].split()
        text = nr_span_text[2].strip()

        if doc is not None:
            start, stop = align_entity_span(int(start), int(stop), doc)

        return cls(int(start), int(stop), label, text, file)

    def __eq__(self, other):
        return (
            self.start == other.start
            and self.stop == other.stop
            and self.label == other.label
            and self.text == other.text
        )

    def __repr__(self):
        return f"Span(text='{self.text}', label='{self.label}', file='{self.file}')"


def extract_spans(
    ann_file_path: Path,
    doc: spacy.tokens.doc.Doc = None,
    ignore_labels: Optional[List[str]] = None,
    remove_parents: bool = False,
    remove_children: bool = False,
    remove_siblings: bool = False,
    verbose: bool = False,
) -> List[Span]:
    """Extracts a list of `Span` objects from a brat ann file

    Parameters
    ----------
    ann_file_path
        Path to the brat ann file
    doc
        If provided, aligns the span with the tokens of the doc
    ignore_labels
        The spans of this label will be ignored
    remove_parents
        If true, remove all parents from the returned spans
    remove_children
        If true, remove all children from the returned spans
    remove_siblings
        If true, remove all siblings from the returned spans
    verbose
        If true, print out the removed spans

    Returns
    -------
    list_of_spans
        List of spans containing their relations
    """
    with ann_file_path.open() as file:
        entities = file.readlines()
    spans = [Span.from_brat(line, ann_file_path.name, doc) for line in entities]
    
    # remove duplicates
    spans_mod = [span for i, span in enumerate(spans) if span not in spans[:i]]
    
    if ignore_labels:
        spans_mod = [span for span in spans_mod if span.label not in ignore_labels]

    for span in spans_mod:
        span.add_relatives(spans_mod)
    if remove_parents:
        spans_mod = [span for span in spans_mod if not span.children]
    if remove_children:
        spans_mod = [span for span in spans_mod if not span.parents]
    if remove_siblings:
        spans_mod = [span for span in spans_mod if not span.siblings]

    if verbose:
        for span in spans:
            if span not in spans_mod:
                print(f"Removed {span}")

    return spans_mod


def brat2doc(
    text_file_path: Path,
    ann_file_path: Path,
    nlp: spacy.lang.es.Language,
    align_offsets: bool = False,
    **kwargs,
) -> spacy.tokens.doc.Doc:
    """Returns BILOU annotations saved in a spacy.tokens.doc.Doc

    Parameters
    ----------
    text_file_path
        Path to the brat txt file
    ann_file_path
        Path to the brat ann file
    nlp
        A spacy Language model used for tokenization and saving the BILOU annotations
    align_offsets
        If true, tries to align the offsets in the brat file with the tokens in the spacy Doc
    **kwargs
        Further kwargs are passed on to the `extract_spans` method

    Returns
    -------
    doc
    """
    text = text_file_path.read_text()
    doc = nlp(text)

    spans = extract_spans(ann_file_path, doc=doc if align_offsets else None, **kwargs)
    doc._.entity_text = [span.text for span in spans]

    tags = spacy.gold.biluo_tags_from_offsets(doc, [span.to_spacy() for span in spans])
    for token, tag in zip(doc, tags):
        token._.ctag = tag

    return doc


def token_at_char_position(doc, char_pos: int) -> spacy.tokens.Token:
    """Returns the token at `char_pos`"""
    token_i = max([t.i for t in doc if t.idx <= char_pos] or [0])
    return doc[token_i]


def align_entity_span(start, stop, doc) -> Tuple[int, int]:
    """Aligns the entity span with the token position in the doc"""
    token_start = token_at_char_position(doc, start)
    token_end = token_at_char_position(doc, stop)

    return token_start.idx, token_end.idx + len(token_end)


def text2df(text_file_path: Path, nlp: spacy.lang.es.Language) -> pd.DataFrame:
    """Returns a data frame containing one sentence per row without BILOU tags.

    Can be used for testing.

    Parameters
    ----------
    text_file_path
        Path to the brat txt file.
    nlp
        A spacy Language model used for tokenization

    Returns
    -------
    dataframe
    """
    text = text_file_path.read_text()
    doc = nlp(text)
    df = doc2df(doc)
    # remove label column since it is meaningless for our purpose
    del df["labels"]

    return df


def doc2df(doc: spacy.tokens.doc.Doc, file: str = None) -> pd.DataFrame:
    """Returns a data frame containing one sentence per row with its BILOU tags.

    Can be used for training/evaluation

    Parameters
    ----------
    doc
        A spacy doc containing the BILOU tags
    file
        The file name the original text belongs to

    Returns
    -------
    dataframe
    """
    data = {
        "text_org": [],
        "text": [],
        "labels": [],
        "file": [],
        "sentence_offset": [],
        "entity_text": [],
    }
    for sentence in doc.sents:
        data["text_org"].append(sentence.text)
        data["text"].append([token.text for token in sentence])
        data["labels"].append([token._.ctag for token in sentence])
        data["sentence_offset"].append(sentence[0].idx)
        data["file"].append(file if file is not None else None)
        if doc._.entity_text is not None:
            entity_texts = [ent_text for ent_text in doc._.entity_text if ent_text in sentence.text]
        else:
            entity_texts = None
        data["entity_text"].append(entity_texts)

    return pd.DataFrame(data)


def doc2ann(doc: spacy.tokens.doc.Doc, predicted_tags: List[str]) -> str:
    """Transforms a doc + predicted tags to a brat ann str

    Parameters
    ----------
    doc
        A spacy Doc containing the original text, used to calculate the offsets
    predicted_tags
        A list of predicted tags coming from a biome.text pipeline

    Returns
    -------
    ann_str
        A string in the brat ann format
    """
    offsets_and_labels = spacy.gold.offsets_from_biluo_tags(doc, predicted_tags)
    ann = ""
    for i, offset_label in enumerate(offsets_and_labels, start=1):
        label, start, stop = offset_label[2], offset_label[0], offset_label[1]
        text = doc.text[start:stop]
        # \n produces invalid ann files!
        text = text.replace("\n", "\\n")
        ann += f"T{i}\t{label} {start} {stop}\t{text}\n"

    return ann


def text2ann(
    text_file_path: Path,
    output_dir: Path,
    nlp: spacy.lang.es.Language,
    pipeline: biome.text.Pipeline,
    replace_tokens: bool = False,
    split_sentences: bool = False,
) -> str:
    """Makes predictions for a brat txt file and writes a brat ann file in the `output_dir`.

    Parameters
    ----------
    text_file_path
        Path to the brat txt file
    output_dir
        Output dir for the brat ann file
    nlp
        A spacy language model
    pipeline
        A biome.text pipeline used to make the predictions
    replace_tokens
        Replaces tokens that were also replaced during our training and evaluation
    split_sentences
        If true, split the longest sentences in half before feeding them to the model in case the sentence is too long
        for the model to be processed.

    Returns
    -------
    ann_str
    """
    text = text_file_path.read_text()
    doc = nlp(text)

    data_df = doc2df(doc)
    if replace_tokens:
        replace_tokens_with_char(data_df, TOKENS_TO_REPLACE_FOR_BERT, "æ")
    while True:
        try:
            predictions = pipeline.predict_batch([{"text": i} for i in data_df.text])
        except WrongValueError as e:
            # Transformer models usually have a limit regarding input tokens, this is a dirty fix
            if split_sentences:
                data_df = split_longest_sentence(data_df)
            else:
                raise e
        else:
            break

    predicted_tags = [tag for prediction in predictions for tag in prediction["tags"]]

    ann_str = doc2ann(doc, predicted_tags)
    file_name = os.path.splitext(text_file_path.name)[0] + ".ann"
    output_dir.mkdir(exist_ok=True)
    (output_dir / file_name).write_text(ann_str)

    return ann_str


def custom_tokenizer(nlp):
    return Tokenizer(
        nlp.vocab,
        prefix_search=spacy.util.compile_prefix_regex(
            tuple(list(nlp.Defaults.prefixes) + [r"-"])
        ).search,
        infix_finditer=spacy.util.compile_infix_regex(
            tuple(list(nlp.Defaults.infixes) + [r"[(:&;\+,]"])
        ).finditer,
        suffix_search=nlp.tokenizer.suffix_search,
    )


def replace_tokens_with_char(df: pd.DataFrame, tokens: List[str], char: str) -> int:
    changes = 0
    for row in df.itertuples():
        idx = [i for i in range(len(row.text)) if row.text[i] in tokens]
        for i in idx:
            row.text[i] = char
            changes += 1

    return changes


def split_longest_sentence(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """Returns a modified dataframe in which the longest sentence is split in two sentences"""
    idx = np.argmax([len(tokens) for tokens in df.text])
    half = int(len(df.iloc[idx].text) / 2)
    first_half_row, second_half_row = df.iloc[idx], df.iloc[idx]
    first_half_row.text, second_half_row.text = first_half_row.text[:half], second_half_row.text[half:]
    if "labels" in df.columns:
        first_half_row.labels, second_half_row.labels = first_half_row.labels[:half], second_half_row.labels[half:]

    modified_df = insert_row(idx+1, df, second_half_row)
    modified_df.iloc[idx] = first_half_row

    if verbose:
        print("Split: ", first_half_row.text, second_half_row.text)

    return modified_df


# Function to insert row in the dataframe
def insert_row(row_number, df, row_value):
    # Starting value of upper half
    start_upper = 0

    # End value of upper half
    end_upper = row_number

    # Start value of lower half
    start_lower = row_number

    # End value of lower half
    end_lower = df.shape[0]

    # Create a list of upper_half index
    upper_half = [*range(start_upper, end_upper, 1)]

    # Create a list of lower_half index
    lower_half = [*range(start_lower, end_lower, 1)]

    # Increment the value of lower half by 1
    lower_half = [x.__add__(1) for x in lower_half]

    # Combine the two lists
    index_ = upper_half + lower_half

    # Update the index of the dataframe
    df.index = index_

    # Insert a row at the end
    df.loc[row_number] = row_value

    # Sort the index labels
    df = df.sort_index()

    return df
