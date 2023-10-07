import re

import nltk

PHONEMIC_REPRS = nltk.corpus.cmudict.dict()
PHONEMIC_REPRS |= {
    "ZIGGED": [["Z", "IH1", "G", "D"]],
    "HOYD": [["HH", "OY1", "D"]],
    "HODE": [["HH", "OW1", "D"]],
    "BINE": [["B", "AY1", "N"]],
    "CRAG": [["K", "R", "AE1", "G"]],
    "BAP": [["B", "AE1", "P"]],
    "TOASTY": [["T", "OW1", "S", "T", "IY0"]],
    "HUDD": [["HH", "AH1", "D"]],
    "zigged": [["Z", "IH1", "G", "D"]],
    "hoyd": [["HH", "OY1", "D"]],
    "hode": [["HH", "OW1", "D"]],
    "bine": [["B", "AY1", "N"]],
    "crag": [["K", "R", "AE1", "G"]],
    "bap": [["B", "AE1", "P"]],
    "toasty": [["T", "OW1", "S", "T", "IY0"]],
    "hudd": [["HH", "AH1", "D"]]
}

ARPABET_VOWELS_TO_IPA = {
    'AA': 'ɑ',
    'AE': 'æ',
    'AH': 'ʌ',
    'AO': 'ɔ',
    'AW': 'aʊ',
    'AX': 'ə',
    'AXR': 'ɚ',
    'AY': 'aɪ',
    'EH': 'ɛ',
    'ER': 'ɝ',
    'EY': 'eɪ',
    'IH': 'ɪ',
    'IX': 'ɨ',
    'IY': 'i',
    'OW': 'oʊ',
    'OY': 'ɔɪ',
    'UH': 'ʊ',
    'UW': 'u',
    'UX': 'ʉ'
}

ARPABET_CONS_TO_IPA = {
    'B': 'b',
    'CH': 'ʧ',
    'D': 'd',
    'DH': 'ð',
    'DX': 'ɾ',
    'EL': 'l',
    'EM': 'm',
    'EN': 'n',
    'F': 'f',
    'G': 'ɡ',
    'HH': 'h',
    'JH': 'ʤ',
    'K': 'k',
    'L': 'l',
    'M': 'm',
    'N': 'n',
    'NG': 'ŋ',
    'P': 'p',
    'Q': 'ʔ',
    'R': 'ɹ',
    'S': 's',
    'SH': 'ʃ',
    'T': 't',
    'TH': 'θ',
    'V': 'v',
    'W': 'w',
    'WH': 'ʍ',
    'Y': 'j',
    'Z': 'z',
    'ZH': 'ʒ',
    'ɣ': 'ɣ'  # FIXME Not ARPABET, but ɣ doesn't exist in ARPABET, and someone annotated a ɣ
    #  (correctly), and this is a quicker hack to fix that. A better method would be to
    #  have a set of all non-ARPABET IPA symbols to check
}

ARPABET_TO_IPA = ARPABET_VOWELS_TO_IPA | ARPABET_CONS_TO_IPA

ARPABET_POSSESSIVE_Z_PHONEMES = {
    'AA', 'AH', 'AW', 'B', 'D', 'EH', 'EY', 'G', 'IH',
    'L', 'N', 'OW',
    'UH', 'V',
    'AE', 'AO', 'AY', 'DH', 'ER', 'IY',
    'M', 'NG', 'OY', 'R', 'UW'
}
ARPABET_POSSESSIVE_IH_Z_PHONEMES = {
    'JH', 'S', 'ZH', 'CH', 'SH', 'Y', 'W', 'Z'
}
ARPABET_POSSESSIVE_S_PHONEMES = {
    'P', 'T', 'F', 'HH', 'K', 'TH'
}

POSSESSIVE_Z_PHONEMES = set()
for phoneme in ARPABET_POSSESSIVE_Z_PHONEMES:
    POSSESSIVE_Z_PHONEMES.add(phoneme)
    POSSESSIVE_Z_PHONEMES.add(ARPABET_TO_IPA[phoneme])
POSSESSIVE_IH_Z_PHONEMES = set()
for phoneme in ARPABET_POSSESSIVE_IH_Z_PHONEMES:
    POSSESSIVE_IH_Z_PHONEMES.add(phoneme)
    POSSESSIVE_IH_Z_PHONEMES.add(ARPABET_TO_IPA[phoneme])
POSSESSIVE_S_PHONEMES = set()
for phoneme in ARPABET_POSSESSIVE_S_PHONEMES:
    POSSESSIVE_S_PHONEMES.add(phoneme)
    POSSESSIVE_S_PHONEMES.add(ARPABET_TO_IPA[phoneme])


def get_phone_dict(filepath=None):
    if filepath is None:
        return PHONEMIC_REPRS

    phone_dict = nltk.defaultdict(list)
    with open(filepath, 'r') as custom_phone_dict_file:
        for line in custom_phone_dict_file.readlines():
            if line.startswith(';;;'):
                continue
            # split on '  ' to get word and then phones
            word, phones = line.split(maxsplit=1)
            word = word.strip()
            phones = phones.split()
            if word.endswith(')'):
                entry_match = re.match(r'(.+)(?:\(\d+\))', word)
                word = entry_match.group(1)

            lower_word = word.lower()
            upper_word = word.upper()
            phone_dict[lower_word].append(phones)
            phone_dict[upper_word].append(phones)
    return phone_dict


def get_phonemic_reprs(raw_word, ipa=False, phone_dict=PHONEMIC_REPRS):
    clean_raw_word = raw_word
    if raw_word.startswith('{'):
        # we have a homophone set, so we only need to look up one of them
        clean_raw_word = raw_word.split(maxsplit=1)[0][1:]
    lowercase_word = clean_raw_word.casefold()
    # we need to get reprs for individual words
    words = lowercase_word.split('-') if '-' in lowercase_word else [lowercase_word]

    phonemic_reprs = []
    for word in words:
        upper_word = word.upper()
        word_not_found = False
        if word in phone_dict:
            phonemic_reprs = _get_phoneme_strings(word, ipa=ipa, phone_dict=phone_dict)
        elif word.endswith("'s") or word.endswith('s'):
            possessive_word = re.sub(r"'", '', word)
            root_word = re.sub(r"'?s$", '', word)
            if possessive_word in phone_dict:
                phonemic_reprs = _get_phoneme_strings(
                    possessive_word, ipa=ipa, phone_dict=phone_dict
                )
            elif root_word in phone_dict:
                phonemic_roots = _get_phoneme_strings(root_word, ipa=ipa, phone_dict=phone_dict)
                phonemic_reprs = [
                    add_possessive_phoneme(
                        phonemic_root, ipa=ipa
                    ) for phonemic_root in phonemic_roots
                ]
            else:
                word_not_found = True
        elif word.endswith("in'") or word.endswith('in'):
            ing_word = re.sub(r"'", '', word) + 'g'
            if ing_word in phone_dict:
                phonemic_reprs = _get_phoneme_strings(ing_word, ipa=ipa, phone_dict=phone_dict)
            else:
                word_not_found = True
        else:
            word_not_found = True

        if word_not_found:
            phonemic_reprs = [upper_word]
    if ipa:
        return [re.sub('-', '', phonemic_repr) for phonemic_repr in phonemic_reprs]

    return phonemic_reprs


def _get_phoneme_strings(word, ipa=False, phone_dict=PHONEMIC_REPRS):
    if isinstance(word, list):
        return '-'.join(word)
    phoneme_reprs = phone_dict[word]
    if ipa:
        phoneme_reprs = [
            [ARPABET_TO_IPA[re.sub(r"\d", '', phoneme)] for phoneme in phonemes]
            for phonemes in phoneme_reprs
        ]

    return ['-'.join(phoneme_repr) for phoneme_repr in phoneme_reprs]


def add_possessive_phoneme(phonemes, ipa=False):
    last_phoneme = re.sub(r"\d+", '', phonemes.rsplit('-', maxsplit=1)[-1])
    if last_phoneme == 'Y' or last_phoneme == 'j' or last_phoneme == 'W' or last_phoneme == 'w':
        print(f"Word ending in Y or W phoneme: {last_phoneme}")

    if last_phoneme in POSSESSIVE_Z_PHONEMES:
        return phonemes + ('-Z' if not ipa else 'z')
    elif last_phoneme in POSSESSIVE_IH_Z_PHONEMES:
        return phonemes + ('-IH0-Z' if not ipa else 'ɪz')
    elif last_phoneme in POSSESSIVE_S_PHONEMES:
        return phonemes + ('-S' if not ipa else 's')
    else:
        raise RuntimeError(f"You missed a phoneme for the possessive form of {phonemes}")


def arpabet_to_ipa(phoneme_str):
    phonemes = phoneme_str.split('-')
    return ''.join([ARPABET_TO_IPA[re.sub(r"\d", '', phoneme)] for phoneme in phonemes])