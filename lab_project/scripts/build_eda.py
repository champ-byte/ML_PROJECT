"""Author, execute, and export the training-only Polars/Plotly EDA notebook."""

import os
from pathlib import Path
import sys
import time

import nbformat
from bs4 import BeautifulSoup
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from textwrap import dedent


PROJECT = Path(__file__).resolve().parents[1]


def build_notebook():
    cells = []

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(dedent(text).strip()))

    def code(text):
        cells.append(nbformat.v4.new_code_cell(dedent(text).strip()))

    md("""
    # Business entity resolution: exploring the training data

    **Purpose:** understand the records, labels, and visible data variation before any algorithm work.
    This notebook uses **Polars** for analysis and **Plotly** for interactive charts.
    Hover for counts, click legend entries to isolate groups, and double-click a chart to reset zoom.

    **Boundary:** only `dataset_small_10k/train/` is opened. The manifest supplies provenance and
    membership of deliberately added records. Validation, evaluation, and competition test records remain unopened.
    No candidate generation, similarity scoring, or model training takes place here.

    Run **Restart Kernel → Run All** to reproduce the results. The final cells write CSV tables and
    `reports/training_eda/findings.md`; the build script also exports an offline HTML report.
    """)
    code("""
    import hashlib
    import json
    from pathlib import Path
    import platform
    import sys
    import polars as pl
    import plotly
    import plotly.graph_objects as go
    from IPython.display import display, Markdown

    candidates = [Path.cwd(), *Path.cwd().parents]
    PROJECT = next((p if (p / 'scripts/eda_helpers.py').exists() else p / 'lab_project')
                   for p in candidates
                   if (p / 'scripts/eda_helpers.py').exists() or (p / 'lab_project/scripts/eda_helpers.py').exists())
    sys.path.insert(0, str(PROJECT / 'scripts'))
    from eda_helpers import Report, bars
    DATA = PROJECT / 'data/dataset_small_10k'
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    report = Report(PROJECT / 'reports/training_eda')
    source_columns = ['entity_id', 'business_name', 'business_address', 'country']
    sources = {
        name: pl.read_csv(DATA / 'train' / f'{name}.tsv', separator='\\t',
                          schema_overrides={column: pl.String for column in source_columns})
        for name in ['source1', 'source2', 'source3']
    }
    truth = pl.read_csv(DATA / 'train/ground_truth.tsv', separator='\\t',
                       schema_overrides={'source1_entity_id': pl.String, 'matched_entity_ids': pl.String})
    truth = truth.with_columns(pl.col('matched_entity_ids').fill_null(''))
    truth = truth.with_columns(
        pl.when(pl.col('matched_entity_ids') == '').then(pl.lit([], dtype=pl.List(pl.String)))
        .otherwise(pl.col('matched_entity_ids').str.split(',')).alias('matches'))
    truth = truth.with_columns(pl.col('matches').list.len().alias('match_count'))
    records = pl.concat([frame.with_columns(pl.lit(name).alias('source')) for name, frame in sources.items()])
    print(f'Python {platform.python_version()} | Polars {pl.__version__} | Plotly {plotly.__version__}')
    """)
    md("""
    ## 1. What did we load, and are the training labels intact?

    Source 1 contains reference businesses. Sources 2 and 3 contain both labeled matches and
    additional distractor records. The counts below describe this constructed training pool.
    Integrity checks compare only training files with their recorded hashes and memberships.
    """)
    code("""
    train_hashes = {}
    for name in [*sources, 'ground_truth']:
        path = DATA / 'train' / f'{name}.tsv'
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == manifest['tsv_sha256'][f'train/{name}.tsv'], f'File changed: {path}'
        train_hashes[name] = actual
    assert records['entity_id'].n_unique() == records.height
    assert truth['source1_entity_id'].n_unique() == truth.height
    assert set(truth['source1_entity_id']) == set(sources['source1']['entity_id'])
    links = truth.select('source1_entity_id', pl.col('matches').alias('target_id')).explode('target_id').drop_nulls('target_id')
    assert links['target_id'].n_unique() == links.height, 'Shared or duplicate target labels'
    target_records = records.filter(pl.col('source') != 'source1')
    assert set(links['target_id']) <= set(target_records['entity_id'])
    train_extras = {key for key, split in manifest['extra_records'].items() if split == 'train'}
    assert train_extras.isdisjoint(set(links['target_id']))
    assert set(target_records['entity_id']) == train_extras | set(links['target_id'])
    assert set(sources['source1']['entity_id']) == {key for key, split in manifest['selected_businesses'].items() if split == 'train'}
    inventory = pl.DataFrame({'table': [*sources, 'ground_truth'],
                             'rows': [frame.height for frame in sources.values()] + [truth.height]})
    report.table('inventory', inventory)
    report.note(f"Loaded {truth.height:,} reference businesses, {links.height:,} labeled match records, and {len(train_extras):,} extra target records. Training hashes and label membership checks passed.")
    """)
    md("""
    ## 2. How are countries represented across sources?

    The stacked chart shows record counts, not numbers of distinct real-world businesses in Sources 2/3.
    Multiple records can belong to one business. Compare sources without assuming each row is a different business.
    """)
    code("""
    composition = records.group_by('source', 'country').len(name='records').sort('source', 'country')
    report.table('source_country_counts', composition)
    report.chart('source-country', bars(composition, 'source', 'records', 'country'),
                 'Training records by source and country', 'Source', 'Records', barmode='stack')
    reference_countries = sources['source1'].group_by('country').len().sort('country')
    report.note('Reference businesses by country: ' + '; '.join(f"{r['country']}: {r['len']:,}" for r in reference_countries.to_dicts()) + '.')
    """)
    md("""
    ## 3. Are names and addresses missing or blank?

    A field is **missing/blank** if it is null, empty, or whitespace-only. We also report these cases
    separately in the CSV. Empty TSV cells parse as nulls, so these counts describe parsed values.
    Missing information is a data-quality observation; we do not fill or change original data.
    """)
    code("""
    missing_parts = []
    for field in ['business_name', 'business_address']:
        missing_parts.append(records.group_by('source', 'country').agg(
            pl.len().alias('records'), pl.col(field).is_null().sum().alias('nulls'),
            (pl.col(field) == '').fill_null(False).sum().alias('empty_strings'),
            ((pl.col(field).str.strip_chars() == '') & (pl.col(field) != '')).fill_null(False).sum().alias('whitespace_only'),
            (pl.col(field).fill_null('').str.strip_chars() == '').sum().alias('missing_blank')
        ).with_columns(pl.lit(field).alias('field')))
    missing = pl.concat(missing_parts).with_columns((100 * pl.col('missing_blank') / pl.col('records')).alias('missing_pct')).sort('source', 'country', 'field')
    report.table('missing_fields', missing, limit=20)
    heat = missing.with_columns((pl.col('source') + ' / ' + pl.col('country')).alias('group')).pivot(on='field', index='group', values='missing_pct').sort('group')
    fields = ['business_name', 'business_address']
    fig = go.Figure(go.Heatmap(x=['Name', 'Address'], y=heat['group'].to_list(), z=heat.select(fields).rows(),
                             colorscale='Blues', zmin=0, zmax=max(1, missing['missing_pct'].max()),
                             colorbar_title='% missing', texttemplate='%{z:.2f}%', hovertemplate='%{y}<br>%{x}: %{z:.2f}% missing<extra></extra>'))
    report.chart('missingness', fig, 'Missing or blank fields by source and country', 'Field', 'Group')
    report.note(f"Across all {records.height:,} records, missing/blank field counts are " + '; '.join(f"{field}: {missing.filter(pl.col('field') == field)['missing_blank'].sum():,}" for field in fields) + '.')
    """)
    md("""
    ## 4. How long are the names and addresses?

    Length means **Unicode characters**, not bytes. Boxes summarize the middle 50%, the line is the median,
    and plotted points show outliers. We do not label a long or short field as incorrect.
    Missing text has length zero here; use the missingness section to distinguish it from present text.
    """)
    code("""
    lengths = records.with_columns(
        pl.col('business_name').fill_null('').str.len_chars().alias('name_chars'),
        pl.col('business_address').fill_null('').str.len_chars().alias('address_chars'))
    stats = lengths.group_by('source', 'country').agg(
        *[expr for field in ['name_chars', 'address_chars'] for expr in [
            pl.col(field).median().alias(field + '_median'),
            pl.col(field).quantile(0.95).alias(field + '_p95'),
            pl.col(field).max().alias(field + '_max')]]).sort('source', 'country')
    report.table('text_length_summary', stats)
    for field, title in [('name_chars', 'Business name'), ('address_chars', 'Business address')]:
        fig = go.Figure()
        for source in sources:
            part = lengths.filter(pl.col('source') == source)
            fig.add_trace(go.Box(y=part[field].to_list(), name=source, boxpoints='outliers'))
        report.chart(field, fig, f'{title} length by source', 'Source', 'Characters')
    report.note(f"Median lengths across the constructed training pool: name {lengths['name_chars'].median():g} characters; address {lengths['address_chars'].median():g} characters. Country/source-specific medians and upper tails are in the summary table.")
    """)
    md("""
    ## 5. How many matches does each reference business have?

    Zero matches is a valid label, not a missing annotation. Counts are per Source 1 business.
    The second chart separates the number of Source 2 and Source 3 matches. These are label properties,
    rather than predictions or results from a matching algorithm.
    """)
    code("""
    reference = truth.join(sources['source1'].select(pl.col('entity_id').alias('source1_entity_id'), 'country'), on='source1_entity_id')
    match_counts = reference.group_by('match_count', 'country').len(name='businesses').sort('match_count', 'country')
    report.table('match_count_distribution', match_counts, limit=30)
    report.chart('match-count', bars(match_counts, 'match_count', 'businesses', 'country'),
                 'True matches per reference business', 'Number of true matches', 'Reference businesses', barmode='stack')
    multiplicity = truth.with_columns(
        pl.col('matches').list.eval(pl.element().str.starts_with('S2-')).list.sum().alias('source2_matches'),
        pl.col('matches').list.eval(pl.element().str.starts_with('S3-')).list.sum().alias('source3_matches'))
    grid = multiplicity.group_by('source2_matches', 'source3_matches').len(name='businesses').sort('source2_matches', 'source3_matches')
    report.table('matches_by_target_source', grid, limit=12)
    xs = sorted(grid['source2_matches'].unique().to_list()); ys = sorted(grid['source3_matches'].unique().to_list())
    lookup = {(r['source2_matches'], r['source3_matches']): r['businesses'] for r in grid.to_dicts()}
    fig = go.Figure(go.Heatmap(x=xs, y=ys, z=[[lookup.get((x,y),0) for x in xs] for y in ys],
                             colorscale='Blues', texttemplate='%{z}', colorbar_title='Businesses'))
    report.chart('match-multiplicity', fig, 'How matches are distributed between Sources 2 and 3', 'Source 2 matches', 'Source 3 matches')
    singletons = truth.filter(pl.col('match_count') == 0).height
    report.note(f"{singletons:,} of {truth.height:,} reference businesses ({100*singletons/truth.height:.2f}%) have no matches. The largest labeled match list contains {truth['match_count'].max()} records.")
    """)
    md("""
    ## 6. What visible text patterns occur?

    We measure non-ASCII characters, digits, and punctuation in each field. Non-ASCII can include
    Devanagari, accents, or symbols: **it is not a language detector or an error flag**.
    The raw columns remain unchanged. Patterns are descriptive indicators, not model features.
    """)
    code("""
    pattern_parts = []
    patterns = {'non_ascii': r'[^\\x00-\\x7F]', 'digit': r'\\d', 'punctuation': r'[^\\p{L}\\p{N}\\s]'}
    for field in fields:
        for label, pattern in patterns.items():
            pattern_parts.append(records.group_by('source', 'country').agg(
                pl.len().alias('records'),
                pl.col(field).fill_null('').str.contains(pattern).sum().alias('with_pattern')
            ).with_columns(pl.lit(field).alias('field'), pl.lit(label).alias('pattern')))
    text_patterns = pl.concat(pattern_parts).with_columns((100 * pl.col('with_pattern') / pl.col('records')).alias('pct')).sort('field','pattern','source','country')
    report.table('text_patterns', text_patterns, limit=12)
    non_ascii = text_patterns.filter(pl.col('pattern') == 'non_ascii').with_columns(
        (pl.col('source') + ' / ' + pl.col('country')).alias('group'))
    report.chart('non-ascii', bars(non_ascii, 'group', 'pct', 'field'),
                 'Records containing non-ASCII characters', 'Source / country', 'Records (%)', barmode='group')
    report.note('Non-ASCII, digit, and punctuation rates are recorded separately for each field, source, and country; these flags alone do not determine text quality or business identity.')
    """)
    md("""
    ## 7. Are field values repeated?

    Count records whose name or address occurs more than once **within the same source and country**.
    Compare raw text with a diagnostic view that only lowercases and trims surrounding whitespace.
    Blank values are excluded. This view does not normalize abbreviations, punctuation, or transliterations.

    A repeated name can belong to different businesses; a repeated address can be shared by many businesses.
    Neither is proof that records should be merged.
    """)
    code("""
    repeats = []
    duplicate_examples = []
    for field in fields:
        for view in ['raw', 'lowercase_trim']:
            expr = pl.col(field) if view == 'raw' else pl.col(field).str.strip_chars().str.to_lowercase()
            values = records.with_columns(expr.alias('value')).filter(pl.col('value').fill_null('').str.strip_chars() != '')
            grouped = values.group_by('source', 'country', 'value').len(name='frequency')
            duplicated = grouped.filter(pl.col('frequency') > 1)
            summary = duplicated.group_by('source','country').agg(
                pl.len().alias('repeated_values'), pl.col('frequency').sum().alias('records_in_repeats'))
            denom = values.group_by('source','country').len(name='nonblank_records')
            repeats.append(denom.join(summary, on=['source','country'], how='left').fill_null(0).with_columns(
                pl.lit(field).alias('field'), pl.lit(view).alias('view'),
                (100 * pl.col('records_in_repeats') / pl.col('nonblank_records')).alias('pct')))
            if view == 'raw':
                duplicate_examples.append(duplicated.sort(['frequency','source','country','value'], descending=[True,False,False,False]).head(8).with_columns(pl.lit(field).alias('field')))
    repetition = pl.concat(repeats).sort('field','source','country','view')
    report.table('repeated_field_rates', repetition, limit=12)
    report.table('repeated_field_examples', pl.concat(duplicate_examples), limit=16)
    plot = repetition.with_columns((pl.col('source') + ' / ' + pl.col('country') + ' / ' + pl.col('field').str.replace('business_','')).alias('group'))
    report.chart('repetition', bars(plot, 'group', 'pct', 'view'),
                 'Records sharing a nonblank field value within source and country', 'Group', 'Nonblank records in repeated groups (%)', barmode='group', height=560)
    report.note('Repeated-field rates use nonblank records as their denominator and stay within source/country groups. They describe text repetition, not duplicate business identities.')
    """)
    md("""
    ## 8. How do labeled matches look side by side?

    Ground truth lets us inspect the same business across sources without inventing matches.
    First measure exact text agreement. Then inspect a reproducible set of examples spread across
    country, target source, and whether both fields agree exactly. This is a small descriptive sample,
    not a random estimate of error prevalence.
    """)
    code("""
    left = sources['source1'].rename({column: 'reference_' + column for column in source_columns})
    right = target_records.rename({**{column: 'target_' + column for column in source_columns}, 'source': 'target_source'})
    paired = links.join(left, left_on='source1_entity_id', right_on='reference_entity_id').join(right, left_on='target_id', right_on='target_entity_id')
    agreement_parts = []
    for field in fields:
        a = pl.col('reference_' + field); b = pl.col('target_' + field)
        nonblank = (a.fill_null('').str.strip_chars() != '') & (b.fill_null('').str.strip_chars() != '')
        category = pl.when(~nonblank).then(pl.lit('missing on either side')).when(a == b).then(pl.lit('same raw text')).when(
            a.str.strip_chars().str.to_lowercase() == b.str.strip_chars().str.to_lowercase()
        ).then(pl.lit('same after lowercase + trim')).otherwise(pl.lit('different text'))
        paired = paired.with_columns(category.alias(field + '_agreement'))
        agreement_parts.append(paired.group_by('target_source', field + '_agreement').len(name='pairs')
            .rename({field + '_agreement': 'agreement'}).with_columns(pl.lit(field).alias('field')))
    agreement = pl.concat(agreement_parts).sort('field','target_source','agreement')
    report.table('labeled_match_exact_agreement', agreement, limit=30)
    agreement_plot = agreement.with_columns((pl.col('target_source') + ' / ' + pl.col('field').str.replace('business_','')).alias('group'))
    report.chart('exact-agreement', bars(agreement_plot, 'group', 'pairs', 'agreement'),
                 'Text agreement within labeled matching pairs', 'Target source / field', 'Labeled pairs', barmode='stack')
    paired = paired.with_columns(((pl.col('business_name_agreement') == 'same raw text') &
                                 (pl.col('business_address_agreement') == 'same raw text')).alias('both_fields_exact'))
    examples = paired.sort('source1_entity_id','target_id').group_by('reference_country','target_source','both_fields_exact', maintain_order=True).head(2)
    example_columns = ['source1_entity_id','target_id','reference_country','target_source','reference_business_name','target_business_name','reference_business_address','target_business_address']
    report.table('labeled_match_examples', examples.select(example_columns), limit=24)
    different_names = paired.filter(pl.col('business_name_agreement') == 'different text').height
    different_addresses = paired.filter(pl.col('business_address_agreement') == 'different text').height
    report.note(f"Among {paired.height:,} labeled pairs, {different_names:,} have different names even after lowercase/trim, and {different_addresses:,} have different addresses after that same limited comparison. These are text differences, not incorrect labels.")
    """)
    md("""
    ## 9. How much of the target pool was deliberately added?

    The manifest identifies extras added during subset preparation. An extra record has no labeled match
    to a selected reference business. It may represent a genuine business elsewhere in the original data.
    Keep this constructed pool composition separate from the original dataset's natural prevalence.
    """)
    code("""
    pool = target_records.with_columns(pl.when(pl.col('entity_id').is_in(sorted(train_extras)))
                                      .then(pl.lit('extra distractor')).otherwise(pl.lit('labeled match')).alias('role'))
    pool_counts = pool.group_by('source','country','role').len(name='records').sort('source','country','role')
    report.table('target_pool_composition', pool_counts)
    pool_plot = pool_counts.with_columns((pl.col('source') + ' / ' + pl.col('country')).alias('group'))
    report.chart('target-pool', bars(pool_plot, 'group', 'records', 'role'),
                 'Constructed training target pool', 'Source / country', 'Records', barmode='stack')
    report.note(f"Extra distractors account for {100 * len(train_extras) / target_records.height:.2f}% of this training target pool. This proportion was chosen during sampling and is not a population estimate.")
    report.finish({'scope': 'reduced training split only', 'python': platform.python_version(),
                   'polars': pl.__version__, 'plotly': plotly.__version__, 'seed_for_subset': manifest['settings']['seed'],
                   'training_hashes': train_hashes, 'reference_businesses': truth.height,
                   'labeled_pairs': links.height, 'extra_records': len(train_extras),
                   'integrity_checks_passed': True})
    display(Markdown('### Findings saved\\nSee `reports/training_eda/findings.md` and the full CSV tables. This completes descriptive EDA; algorithm design and model training remain outside this notebook.'))
    """)
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3 (ML lab uv environment)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]}})


def main():
    notebooks = PROJECT / "notebooks"
    reports = PROJECT / "reports/training_eda"
    notebooks.mkdir(exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    runtime = PROJECT / ".jupyter_runtime"
    runtime.mkdir(exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    notebook = build_notebook()
    path = notebooks / "01_training_eda.ipynb"
    nbformat.write(notebook, path)
    started = time.perf_counter()
    print("Executing training-only EDA notebook...", flush=True)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    try:
        NotebookClient(notebook, km=manager, timeout=180, startup_timeout=60,
                       resources={"metadata": {"path": str(PROJECT)}}).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    exporter = HTMLExporter()
    exporter.exclude_input = True
    body, _ = exporter.from_notebook_node(notebook)
    # The plain-HTML Plotly outputs embed their runtime. This notebook uses no
    # MathJax or RequireJS; remove nbconvert's unused CDN script tags for offline use.
    document = BeautifulSoup(body, "html.parser")
    for script in document.find_all("script", src=True):
        script.decompose()
    (reports / "training_eda.html").write_text(str(document), encoding="utf-8")
    print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells in {time.perf_counter() - started:.1f}s")
    print(f"Notebook: {path}\nOffline report: {reports / 'training_eda.html'}")


if __name__ == "__main__":
    main()
