import os
import re
import sys
import textwrap
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

import click as click
import yaml
from pkg_resources._vendor.more_itertools import pairwise
from praatio import textgrid
from praatio.data_classes.interval_tier import IntervalTier
from praatio.utilities.constants import Interval
from tqdm import tqdm

from common_main_methods import get_input_filepaths, std_out_err_redirect_tqdm
from phonemic import get_phonemic_reprs, arpabet_to_ipa, ARPABET_TO_IPA, get_phone_dict

SILENCE_MARKERS_WORDS = frozenset(['{SL}', 'sp', '{LG}', '{BR}'])
SILENCE_MARKERS_PHONES = frozenset(['sp', 'sil'])


@click.command()
@click.option(
    '--pra-inputs-dir-path',
    prompt='Enter the directory that contains the sclite diffs between the reference transcripts'
           ' and the hypothesis transcripts\n',
    help=textwrap.dedent(
        '''\
        All filenames in the directory should be in the filename format from the PNWE Bias in ASR
         research group, which is as follows:

        {City Code}{Neighborhood Code}{Unique ID Number}{Ethnicity}{Sex}{Generation}{Family Code}#{Task Code}_1/{hypothesis_system}_hyp.trn.pra
        e.g., EDP74CF1T#RP_1/google_hyp.trn.pra
        
        The extension should be .pra, which is output by sclite's TRN to TRN evaluation.
        See documentation at https://github.com/usnistgov/SCTK/blob/master/doc/sclite.htm
        \n
        '''
    )
)
@click.option(
    '--textgrid-inputs-dir-path',
    prompt='Enter the directory that contains the Praat TextGrids with phones, words, markers, and'
           ' hyp tiers\n',
    help=textwrap.dedent(
            '''\
            The directory of Praat TextGrids which must have the following tiers:
            * `phone`
            * `word`
            * `markers`
            * `{system}-hyp`, e.g., `google-hyp`

            All filenames in the directory should be in the filename format from the PNWE Bias in ASR
             research group, which is as follows:

            {City Code}{Neighborhood Code}{Unique ID Number}{Ethnicity}{Sex}{Generation}{Family Code}_{Task Code}/{Speaker-Task Identifier}.TextGrid
            e.g., EDP74CF1T_RP/EDP74CF1T_RP.TextGrid
            
            NB: This system assumes that the following tokens should be ignored:
            {'{SL}', 'sp', '{LG}', '{BR}', 'sil'}
            \n
            '''
    )
)
@click.option(
    '--wav-inputs-dir-path',
    prompt='Enter the directory that contains the reference WAVE files\n',
    help=textwrap.dedent(
        '''\

        All filenames in the directory should be in the filename format from the PNWE Bias in ASR
         research group, which is as follows:

        {City Code}{Neighborhood Code}{Unique ID Number}{Ethnicity}{Sex}{Generation}{Family Code}_{Task Code}/{Speaker-Task Identifier}.wav
        e.g., EDP74CF1T_RP/EDP74CF1T_RP.wav
        \n
        '''
    )
)
@click.option(
    '--rules-input-path',
    prompt='Enter the filepath of the YAML-formatted regexp rules used to identify phonetic'
           ' markers\n',
    help=textwrap.dedent(
        '''\
        The filepath of the YAML-formatted regexp rules used to identify phonetic markers.
        See example under marker_rules/mkscott_thesis_rules.yaml
        \n
        '''
    )
)
@click.option(
    '--pronunciation-dict-path', default=None,
    help=textwrap.dedent(
        '''\
        The path for the CMUdict-formatted canonical pronunciation dictionary. If no filepath is
        provided, CMUdict from NLTK will be used by default.
        \n
        '''
    )
)
@click.option(
    '--output-dir-path', prompt='Enter the path to output files to\n',
    help=textwrap.dedent(
        '''\
        The directory path to output modified TextGrids.
        \n
        '''
    )
)
def analyzer_main(
        pra_inputs_dir_path, textgrid_inputs_dir_path, wav_inputs_dir_path, rules_input_path,
        pronunciation_dict_path, output_dir_path
):
    """
    Analysis program that takes input Praat TextGrids of transcribed speech, sclite diff outputs
     between those TextGrids and hypothesis transcriptions, and a set of phonetic marker
     identification rules, and outputs new TextGrids that include possible identified phonetic
     markers and time-aligned hypothesis tiers.
    """
    phone_dict = get_phone_dict(pronunciation_dict_path)

    symlink_root = make_input_symlinks(
        pra_inputs_dir_path, textgrid_inputs_dir_path, wav_inputs_dir_path, output_dir_path
    )

    aligned_error_dict = {}
    datetime_str = datetime.now().strftime('%d_%b_%y_%H-%M-%S%Z')
    dirpath, dirnames, filenames = next(os.walk(symlink_root))
    with std_out_err_redirect_tqdm() as orig_stdout:
        print("Aligning sclite error outputs with TextGrids...")
        for dirname in tqdm(dirnames, file=orig_stdout, dynamic_ncols=True):
            aligned_error_dict |= get_all_errors(f"{dirpath}/{dirname}", datetime_str)

    hyp_textgrids_dirpath = f"{output_dir_path}/hyp_textgrids_{datetime_str}"
    find_and_add_marker_candidates(hyp_textgrids_dirpath, rules_input_path, phone_dict)


def make_input_symlinks(pra_dir, textgrid_dir, wav_dir, output_dir_path):
    symlink_output_root = f"{output_dir_path}/symlinks"
    pra_filepaths = get_input_filepaths(pra_dir, ['pra'])
    textgrid_filepaths = get_input_filepaths(textgrid_dir, ['textgrid'])
    wav_filepaths = get_input_filepaths(wav_dir, ['wav'])

    filepath_dict = defaultdict(set)
    # pra files
    for filepath in pra_filepaths:
        _, spkr_task_id, filename = filepath.rsplit('/', maxsplit=2)

        # drop the extraneous `_{num}` and change the '#' to an '_'
        spkr_task_id = spkr_task_id.replace('#', '_').rsplit('_', maxsplit=1)[0]
        filepath_dict[spkr_task_id].add(filepath)

    # TextGrid files
    for filepath in textgrid_filepaths:
        spkr_task_id = filepath.rsplit('/', maxsplit=1)[-1]
        spkr_task_id = spkr_task_id.split('.')[0]
        filepath_dict[spkr_task_id].add(filepath)

    # wav files
    for filepath in wav_filepaths:
        spkr_task_id = filepath.rsplit('/', maxsplit=1)[-1]
        spkr_task_id = spkr_task_id.split('.')[0]
        filepath_dict[spkr_task_id].add(filepath)

    for spkr_task_id, filepaths in filepath_dict.items():
        spkr_task_output_dir = f"{symlink_output_root}/{spkr_task_id}"
        try:
            os.makedirs(spkr_task_output_dir)
        except FileExistsError:
            pass

        for filepath in filepaths:
            path = Path(filepath)
            filename, ext = path.name, path.suffix
            new_filename = spkr_task_id
            if ext == '.pra':
                system = filename.split('_', maxsplit=1)[0]
                new_filename = f"{system}-{new_filename}"
            try:
                os.symlink(filepath, f"{spkr_task_output_dir}/{new_filename}{ext}")
            except FileExistsError:
                pass  # assumption is that if the file already exists, it's the one you want

    return symlink_output_root


def get_all_errors(input_spkr_dir, datetime_str):
    spkr_tsk_id = input_spkr_dir.rsplit('/', maxsplit=1)[-1]
    systems_to_err_objs = {}
    wav_filepath = _get_and_assert_one_file(input_spkr_dir, 'wav')
    tg_filepath = _get_and_assert_one_file(input_spkr_dir, 'textgrid')
    error_counts = Counter()
    for input_filepath in get_input_filepaths(input_spkr_dir, ['pra']):
        system = input_filepath.rsplit('/', maxsplit=1)[-1].rsplit('-', maxsplit=1)[0]

        with open(input_filepath, 'r') as input_file:
            lines_to_eval = []
            for line in input_file.readlines():
                if re.match(r"^(?:>> )?(?:REF|HYP).*$", line):
                    lines_to_eval.append(line)
            # create pairs of ref & hyp
            lines_to_eval = zip(*[iter(lines_to_eval)]*2)

            aligned_pairs = []
            for ref_line, hyp_line in lines_to_eval:
                ref_words = re.sub(r"(?:>> )?REF:", '', ref_line).split()
                hyp_words = re.sub(r"(?:>> )?HYP:", '', hyp_line).split()
                aligned_pairs.extend(list(zip(ref_words, hyp_words)))

            error_objs = []
            for idx, aligned_pair in enumerate(aligned_pairs):
                ref_word, hyp_word = aligned_pair
                error_obj = {
                    'ref': ref_word, 'hyp': hyp_word, 'index': idx
                }
                if re.match(r"^\*+$", ref_word):
                    # Insertion
                    error_obj['error'] = 'ins'
                    error_counts['ins'] += 1
                elif re.match(r"^\*+$", hyp_word):
                    # Deletion
                    error_obj['error'] = 'del'
                    error_counts['del'] += 1
                elif ref_word.isupper and hyp_word.isupper():
                    # Substitution
                    error_obj['error'] = 'sub'
                    error_counts['sub'] += 1
                else:
                    # Correct
                    error_obj['error'] = 'corr'
                error_objs.append(error_obj)

            error_objs = _add_alignments(error_objs, tg_filepath, system, datetime_str)
            systems_to_err_objs[system] = error_objs
    return {
        spkr_tsk_id: {
            'wav': wav_filepath, 'textgrid': tg_filepath,
            'error_intervals': systems_to_err_objs,
            'error_counts': error_counts
        }
    }


def _get_and_assert_one_file(dir_path, ext):
    filepaths = list(get_input_filepaths(dir_path, [ext]))
    if len(filepaths) > 1:
        raise RuntimeError(
            f"Expected only 1 .{ext} file in {dir_path}, but found {len(filepaths)}:"
            f"\n{filepaths}"
        )
    return filepaths[0]


def _add_alignments(error_objs, tg_filepath, system, datetime_str):
    textgrid_obj = textgrid.openTextgrid(tg_filepath, includeEmptyIntervals=True)
    filtered_tg_intervals = []
    for interval in textgrid_obj.getTier('word').entries:
        word = interval.label
        # TODO configure words to ignore in textgrid
        if not word or word in SILENCE_MARKERS_WORDS:
            continue
        filtered_tg_intervals.append(interval)

    # combined means that ins-errors were combined into other adjacent errors
    combined_err_objs = []
    ins_err_seq = []
    last_non_ins_err_obj = None
    for error_obj in error_objs:
        combined_err_obj = error_obj.copy()
        if error_obj['error'] == 'ins':
            ins_err_seq.append(combined_err_obj['hyp'].upper())
            continue
        elif len(ins_err_seq) > 0:
            hyp_word = combined_err_obj['hyp']
            hyp_word = hyp_word.lower() if combined_err_obj['error'] == 'corr' else hyp_word.upper()
            combined_err_obj['hyp'] = f"ins: [{' '.join(ins_err_seq)}] {hyp_word}"
            ins_err_seq = []
        else:
            last_non_ins_err_obj = combined_err_obj
        combined_err_objs.append(combined_err_obj)

    # if the last sequence is all ins-errors, then we have some leftover we need to deal with
    if len(ins_err_seq) > 0:
        hyp_word = last_non_ins_err_obj['hyp']
        hyp_word = hyp_word.lower() if last_non_ins_err_obj['error'] == 'corr' else hyp_word.upper()
        last_non_ins_err_obj['hyp'] = f"{hyp_word} ins: [{' '.join(ins_err_seq)}]"

    if len(filtered_tg_intervals) != len(combined_err_objs):
        print(
            f"Textgrid at {tg_filepath}"
            f"\n does not have the same number of intervals as given alignment file excluding"
            f" insertion errors."
            f"\n\tExpected: {len(filtered_tg_intervals)}"
            f"\n\tActual: {len(combined_err_objs)}"
            f"\nPlease ensure this is using the same TextGrid used to generate the original"
            f" ref file.",
            file=sys.stderr
        )
        return error_objs
    new_error_objs = []
    offset_in_sec = textgrid_obj.minTimestamp

    tg_path_obj = Path(tg_filepath)
    hyp_textgrids_dirpath = f"{tg_path_obj.parent.parent.parent}/hyp_textgrids_{datetime_str}"

    try:
        os.makedirs(hyp_textgrids_dirpath)
    except FileExistsError:
        pass

    new_textgrid_filepath = f"{hyp_textgrids_dirpath}/{tg_path_obj.stem}_hyp{tg_path_obj.suffix}"
    try:
        new_textgrid_file = open(new_textgrid_filepath, 'x')
        new_textgrid = textgrid_obj.new()
        new_textgrid = new_textgrid.editTimestamps(-1 * offset_in_sec, reportingMode="silence")
        new_textgrid = _fix_textgrid_boundaries(new_textgrid, -1 * offset_in_sec)
        new_textgrid.save(
            new_textgrid_filepath, "long_textgrid", includeBlankSpaces=True,
            reportingMode="silence"
        )
        new_textgrid_file.close()
    except FileExistsError:
        new_textgrid = textgrid.openTextgrid(new_textgrid_filepath, includeEmptyIntervals=True)

    sys_hyp_tier_name = f'{system}-hyp'

    try:
        new_textgrid_hyp_tier = new_textgrid.getTier(sys_hyp_tier_name)
    except KeyError:
        new_textgrid_hyp_tier = None
    if not new_textgrid_hyp_tier:
        new_textgrid_hyp_tier = IntervalTier(
                sys_hyp_tier_name, [], new_textgrid.minTimestamp, new_textgrid.maxTimestamp
        )
        new_textgrid.addTier(new_textgrid_hyp_tier, reportingMode="error")

    for error_obj, tg_interval in zip(combined_err_objs, filtered_tg_intervals):
        if error_obj['ref'].casefold() != tg_interval.label.casefold():
            print(
                f"TextGrid not aligned with pra file:"
                f"\n\tTextGrid Path: {tg_filepath}"
                f"\n\tTextGrid Interval:{tg_interval}"
                f"\n\tExpected Token: {error_obj['ref']}",
                file=sys.stderr
            )
            continue

        new_error_obj = error_obj.copy()
        new_error_obj['start'] = tg_interval.start - offset_in_sec
        new_error_obj['end'] = tg_interval.end - offset_in_sec
        new_error_objs.append(new_error_obj)

        new_tg_interval = Interval(
            new_error_obj['start'], new_error_obj['end'], new_error_obj['hyp']
        )
        new_textgrid_hyp_tier.insertEntry(new_tg_interval)

    if len(new_textgrid_hyp_tier.entries) >= len(
            new_textgrid.getTier(sys_hyp_tier_name).entries
    ):
        new_textgrid.replaceTier(sys_hyp_tier_name, new_textgrid_hyp_tier, reportingMode="error")
        new_textgrid.save(
            new_textgrid_filepath, "long_textgrid", includeBlankSpaces=True, reportingMode="error"
        )

    return new_error_objs


def _fix_textgrid_boundaries(textgrid_obj, offset):
    new_textgrid_obj = textgrid_obj.new()
    new_textgrid_obj.maxTimestamp += offset
    for tier_name, interval_tier in zip(textgrid_obj.tierNames, textgrid_obj.tiers):
        new_intervals = []
        for interval1, interval2 in pairwise(interval_tier.entries):
            if interval1.end > interval2.start:
                new_interval1 = Interval(interval1.start, interval2.start, interval1.label)
            else:
                new_interval1 = interval1
            new_intervals.append(new_interval1)
        new_intervals.append(interval_tier.entries[-1])
        new_interval_tier = IntervalTier(
            tier_name, new_intervals, interval_tier.minTimestamp,
            interval_tier.maxTimestamp + offset
        )
        new_textgrid_obj.replaceTier(tier_name, new_interval_tier, reportingMode='error')
    return new_textgrid_obj


def find_and_add_marker_candidates(tg_dirpath, rules_filepath, phone_dict):
    with open(rules_filepath, 'r+b') as rules_file:
        rules_yaml = yaml.safe_load(rules_file.read())

    print("Analyzing rules for each file...")
    with std_out_err_redirect_tqdm() as orig_stdout:
        filepaths = list(get_input_filepaths(tg_dirpath, acceptable_exts=['textgrid']))
        for filepath in tqdm(filepaths, file=orig_stdout, dynamic_ncols=True):
            tg = textgrid.openTextgrid(filepath, includeEmptyIntervals=True)

            marker_tg_intervals = []
            possible_marker_tg_intervals = []
            for tg_interval in tg.getTier('word').entries:
                if tg_interval.label in SILENCE_MARKERS_WORDS:
                    continue

                ref_word = tg_interval.label
                if not re.match(r'\w+', ref_word):
                    continue

                ref_transcribed_phones = _get_phones_from_tg(tg, tg_interval)
                if not ref_transcribed_phones:
                    continue

                canon_phonemic_reprs = get_phonemic_reprs(
                    ref_word, ipa=True, phone_dict=phone_dict
                )
                matched_markers = set()
                possible_markers = set()
                for rule_name, rule_list in rules_yaml.items():
                    for rule_entry in rule_list:
                        if rule_name in matched_markers:
                            break  # if we've already matched, no need to keep going

                        ref_rule_str, canon_rule_str = rule_entry['ref'], rule_entry['canon']
                        ref_rule = parse_rule_to_regex(ref_rule_str)
                        canon_rule = parse_rule_to_regex(canon_rule_str)

                        has_a_match = False
                        for phonemic_repr in canon_phonemic_reprs:
                            has_a_match |= bool(re.match(canon_rule, phonemic_repr))
                        if not has_a_match:
                            continue  # All rules must match for this to be a candidate

                        possible_markers.add(rule_name)

                        # all rules must match for this to be a candidate
                        if not re.match(ref_rule, ref_transcribed_phones):
                            # the hand-annotated phones don't match the rule, so this isn't a
                            #  candidate
                            continue

                        matched_markers.add(rule_name)

                marker_tg_intervals.append(
                    Interval(
                        tg_interval.start, tg_interval.end, f"{' '.join(matched_markers)}"
                    )
                )
                possible_marker_tg_intervals.append(
                    Interval(
                        tg_interval.start, tg_interval.end, f"{' '.join(possible_markers)}"
                    )
                )

            # create and add tiers
            markers_interval_tier = IntervalTier(
                'markers', marker_tg_intervals,
                minT=tg.minTimestamp, maxT=tg.maxTimestamp
            )
            possible_markers_interval_tier = IntervalTier(
                'poss-markers', possible_marker_tg_intervals,
                minT=tg.minTimestamp, maxT=tg.maxTimestamp
            )

            # tierIndex 0 = 'word', tierIndex 1 = 'phone'
            tg.addTier(markers_interval_tier, tierIndex=2)
            tg.addTier(possible_markers_interval_tier, tierIndex=3)
            tg.save(filepath, "long_textgrid", includeBlankSpaces=True, reportingMode="error")


def _get_phones_from_tg(textgrid_obj, tg_interval):
    phone_tier = textgrid_obj.getTier('phone')
    tmp_tg_tier = IntervalTier(
        'tmp-word', [tg_interval], phone_tier.minTimestamp, phone_tier.maxTimestamp
    )
    target_word = tg_interval.label
    if not target_word or target_word in SILENCE_MARKERS_WORDS:
        return ''
    intersection_tier = phone_tier.intersection(tmp_tg_tier)
    phone_list = []
    for interval in intersection_tier.entries:
        phone, word = interval.label.split('-')
        # this shouldn't happen at all, but I have seen it occasionally, perhaps  because we have
        #  imperfectly aligned intervals
        if not phone or phone in SILENCE_MARKERS_PHONES:
            continue
        if word.casefold() == target_word.casefold():
            phone_list.append(phone)
        else:
            print(
                f"Unexpected overlap in phone tier and word tier:"
                f"\n\tTarget Word:{target_word}"
                f"\n\tFound: {intersection_tier.entries}",
                file=sys.stderr
            )

    return arpabet_to_ipa('-'.join(phone_list))


ALL_IPA_GROUP = f"[{''.join(set(ARPABET_TO_IPA.values()))}]"


def parse_rule_to_regex(rule_str):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    rule = rule_str.replace('(?#all_ipa)', ALL_IPA_GROUP)
    to_replace_groups = re.findall(fr"(\(\?#\^({ALL_IPA_GROUP}+)\))", rule)
    if to_replace_groups:
        for to_replace, ipa_chars in to_replace_groups:
            filtered_ipa_str = re.sub(fr"[{ipa_chars}]", '', ALL_IPA_GROUP)
            rule = rule.replace(to_replace, filtered_ipa_str)
    return rule


if __name__ == '__main__':
    analyzer_main()
