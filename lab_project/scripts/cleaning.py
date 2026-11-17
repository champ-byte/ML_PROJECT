"""Conservative, field-aware retrieval views; raw dataset values are never edited."""

from collections import Counter
from functools import partial
import re
import unicodedata

import polars as pl

from blocking import normalize_text


VARIANTS = ('baseline', 'placeholders', 'placeholders_abbreviations')
FIELDS = ('business_name', 'business_address')
ADDRESS_COMPONENTS = frozenset({'null', 'n/a', '<null>'})
NAME_ABBREVIATIONS = {'pvt': 'private', 'ltd': 'limited', 'inc': 'incorporated', 'corp': 'corporation'}
ADDRESS_ABBREVIATIONS = {'rd': 'road', 'ave': 'avenue', 'blvd': 'boulevard',
                         'apt': 'apartment', 'ste': 'suite'}


def placeholder_text(value, field):
    text = unicodedata.normalize('NFKC', value or '')
    if field == 'business_address':
        # Bare null/N/A are removed ONLY as whole comma-delimited components.
        # Null Road / N/A Building and business names remain intact.
        pieces = [piece for piece in text.split(',') if piece.strip().casefold() not in ADDRESS_COMPONENTS]
        text = ','.join(pieces)
        # Explicit angle-bracket NULL may also be embedded in useful text.
        text = re.sub(r'<null>', ' ', text, flags=re.I)
    elif field == 'business_name':
        text = re.sub(r'<city_name>', ' ', text, flags=re.I)
    return text


def clean_for_retrieval(value, field, variant='placeholders'):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown cleaning variant: {variant}')
    if field not in FIELDS:
        raise ValueError(f'Cleaning is restricted to name/address fields: {field}')
    if variant == 'baseline':
        return normalize_text(value)
    text = normalize_text(placeholder_text(value, field))
    if variant == 'placeholders_abbreviations':
        mapping = NAME_ABBREVIATIONS if field == 'business_name' else ADDRESS_ABBREVIATIONS
        text = ' '.join(mapping.get(token, token) for token in text.split())
    return text


def normalizer(variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown cleaning variant: {variant}')
    return partial(clean_for_retrieval, variant=variant)


def rules():
    return {'rule_source': 'training-only audit; fixed before validation comparison',
            'variants': list(VARIANTS),
            'address_whole_comma_components': sorted(ADDRESS_COMPONENTS),
            'address_explicit_embedded_marker': '<NULL> (case-insensitive)',
            'name_explicit_marker': '<CITY_NAME> (case-insensitive)',
            'name_abbreviations': NAME_ABBREVIATIONS, 'address_abbreviations': ADDRESS_ABBREVIATIONS,
            'preserved': ['raw fields', 'country', 'Null Industries and N/A Science Inc',
                          'Null Road and N/A Building', 'accents', 'non-Latin text',
                          'Unicode joiners/non-joiners', 'ambiguous St/No', 'legal suffix content'],
            'excluded': ['external lookups', 'transliteration', 'typo correction',
                         'word deletion for generic company suffixes', 'validation-derived rules']}


def audit(sources):
    counts, denominators, examples, components, formats, protected = Counter(), Counter(), [], Counter(), Counter(), []
    examples_seen = Counter()
    issues = ('missing_blank', 'irregular_whitespace', 'unicode_format_character', 'nfkc_changes',
              'placeholder_cleanup_changes_view', 'abbreviation_expansion_changes_view')
    for source, frame in sources.items():
        for row in frame.sort('entity_id').to_dicts():
            for field in FIELDS:
                value = row[field] or ''
                key = (source, row['country'], field)
                denominators[key] += 1
                baseline = clean_for_retrieval(value, field, 'baseline')
                placeholder = clean_for_retrieval(value, field, 'placeholders')
                expanded = clean_for_retrieval(value, field, 'placeholders_abbreviations')
                flags = {'missing_blank': not value.strip(),
                         'irregular_whitespace': bool(re.search(r'\s{2,}|^\s|\s$', value)),
                         'unicode_format_character': any(unicodedata.category(char) == 'Cf' for char in value),
                         'nfkc_changes': value != unicodedata.normalize('NFKC', value),
                         'placeholder_cleanup_changes_view': baseline != placeholder,
                         'abbreviation_expansion_changes_view': placeholder != expanded}
                for issue, present in flags.items():
                    counts[(*key, issue)] += int(present)
                    example_key = (*key, issue)
                    if present and examples_seen[example_key] < 3:
                        examples_seen[example_key] += 1
                        examples.append({'entity_id': row['entity_id'], 'source': source, 'country': row['country'],
                                         'field': field, 'issue': issue, 'raw': value, 'baseline': baseline,
                                         'placeholders': placeholder, 'placeholders_abbreviations': expanded})
                if field == 'business_address':
                    for piece in value.split(','):
                        if piece.strip().casefold() in ADDRESS_COMPONENTS:
                            components[(source, row['country'], piece.strip())] += 1
                if field == 'business_name' and re.search(r'\bnull\b|\bn/a\b', value, re.I):
                    protected.append({'entity_id': row['entity_id'], 'raw_name': value,
                                      'placeholder_view': placeholder, 'source': source, 'country': row['country']})
                for char in value:
                    if unicodedata.category(char) == 'Cf':
                        formats[(source, row['country'], field, f'U+{ord(char):04X}', unicodedata.name(char,'UNKNOWN'))] += 1
    summaries = [{'source': source, 'country': country, 'field': field, 'issue': issue,
                  'records': denominators[(source,country,field)],
                  'affected_records': counts[(source,country,field,issue)],
                  'affected_pct': 100*counts[(source,country,field,issue)]/denominators[(source,country,field)]}
                 for source,country,field in sorted(denominators) for issue in issues]
    component_rows = [{'source': s, 'country': c, 'component': v, 'occurrences': n} for (s,c,v),n in sorted(components.items())]
    format_rows = [{'source': s, 'country': c, 'field': f, 'codepoint': code, 'name': name, 'occurrences': n}
                   for (s,c,f,code,name),n in sorted(formats.items())]
    return {'counts': pl.DataFrame(summaries),
            'examples': pl.DataFrame(examples),
            'address_components': pl.DataFrame(component_rows, schema={'source':pl.String,'country':pl.String,'component':pl.String,'occurrences':pl.UInt32}),
            'format_characters': pl.DataFrame(format_rows, schema={'source':pl.String,'country':pl.String,'field':pl.String,'codepoint':pl.String,'name':pl.String,'occurrences':pl.UInt32}),
            'protected_names': pl.DataFrame(protected, schema={'entity_id':pl.String,'raw_name':pl.String,'placeholder_view':pl.String,'source':pl.String,'country':pl.String})}
