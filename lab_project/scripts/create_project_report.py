"""Create the academic project PDF from saved evidence (ReportLab + pypdf)."""

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.graphics.shapes import Drawing, Line, Rect, String, Circle, PolyLine
from pypdf import PdfReader


ROOT=Path(__file__).resolve().parents[2]
LAB=ROOT/'lab_project';OUT=ROOT/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
NAVY=colors.HexColor('#17324d');TEAL=colors.HexColor('#087f8c');BLUE=colors.HexColor('#3174b5')
GRAY=colors.HexColor('#536174');LIGHT=colors.HexColor('#edf3f7');GREEN=colors.HexColor('#23846b')
INK=colors.HexColor('#202e3c');RED=colors.HexColor('#bd5050')
W,H=A4;M=46;CW=W-2*M


def load_json(path):return json.loads(path.read_text(encoding='utf-8'))
def read_csv(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))
def pct(value,d=2):return f'{100*float(value):.{d}f}%'


for name,file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf'),('Arial-Italic','ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-Bold')
STYLES={
 'body':ParagraphStyle('body',fontName='Arial',fontSize=10.2,leading=15,textColor=INK,spaceAfter=9),
 'small':ParagraphStyle('small',fontName='Arial',fontSize=8.4,leading=12,textColor=GRAY,spaceAfter=7),
 'h1':ParagraphStyle('h1',fontName='Arial-Bold',fontSize=21,leading=27,textColor=NAVY,spaceAfter=15),
 'h2':ParagraphStyle('h2',fontName='Arial-Bold',fontSize=12.4,leading=17,textColor=TEAL,spaceBefore=10,spaceAfter=8),
 'cell':ParagraphStyle('cell',fontName='Arial',fontSize=8.9,leading=12,textColor=INK),
 'head':ParagraphStyle('head',fontName='Arial-Bold',fontSize=8.8,leading=12,textColor=colors.white),
 'cover':ParagraphStyle('cover',fontName='Arial-Bold',fontSize=33,leading=41,textColor=NAVY,spaceAfter=17),
 'subtitle':ParagraphStyle('subtitle',fontName='Arial',fontSize=14,leading=21,textColor=GRAY,spaceAfter=12),
 'caption':ParagraphStyle('caption',fontName='Arial-Italic',fontSize=8.3,leading=11.5,textColor=GRAY,spaceAfter=12),
 'formula':ParagraphStyle('formula',fontName='Arial',fontSize=10.3,leading=16,textColor=NAVY,spaceAfter=9),
}


def p(text,style='body'):return Paragraph(text,STYLES[style])
story=[]
def para(text,style='body'):story.append(p(text,style))
def h2(text):para(text,'h2')
def section(number,title):
    story.append(PageBreak());para(f'{number}. {title}','h1')
def table(headers,rows,widths=None):
    data=[[p(escape(str(value)),'head') for value in headers]]
    data += [[p(escape(str(value)),'cell') for value in row] for row in rows]
    result=Table(data,colWidths=widths or [CW/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
    result.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
        ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),
        ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#ccd7e1'))]))
    story.extend([result,Spacer(1,10)])
def callout(text):
    box=Table([[p(text)]],colWidths=[CW]);box.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),LIGHT),('LINEBEFORE',(0,0),(0,0),3,TEAL),
        ('LEFTPADDING',(0,0),(-1,-1),13),('RIGHTPADDING',(0,0),(-1,-1),12),
        ('TOPPADDING',(0,0),(-1,-1),11),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.extend([box,Spacer(1,12)])


def bar_chart(labels,values,title,low=0,high=1,format_value=None,color_list=None):
    drawing=Drawing(CW,210);x0=49;y0=45;plot_w=CW-65;plot_h=128
    drawing.add(String(0,196,title,fontName='Arial-Bold',fontSize=10.5,fillColor=NAVY))
    tick_count=3 if low or high>1 else 4
    for i in range(tick_count+1):
        value=low+(high-low)*i/tick_count;y=y0+plot_h*i/tick_count
        drawing.add(Line(x0,y,x0+plot_w,y,strokeColor=colors.HexColor('#dce4eb'),strokeWidth=.5))
        label=f'{value:.3f}' if low else f'{value:.0%}' if high<=1 else f'{value:.0f}'
        drawing.add(String(x0-8,y-3,label,textAnchor='end',fontName='Arial',fontSize=7.5,fillColor=GRAY))
    for i,(label,value) in enumerate(zip(labels,values)):
        step=plot_w/len(labels);x=x0+i*step+step*.2;height=plot_h*(value-low)/(high-low)
        drawing.add(Rect(x,y0,step*.6,max(0,height),fillColor=(color_list or [TEAL]*len(labels))[i],strokeColor=None))
        value_label=format_value(value) if format_value else f'{value:.4f}'
        drawing.add(String(x+step*.3,y0+max(0,height)+6,value_label,textAnchor='middle',fontName='Arial-Bold',fontSize=7.8,fillColor=INK))
        for line_num,line in enumerate(label.split('\n')):
            drawing.add(String(x+step*.3,y0-16-10*line_num,line,textAnchor='middle',fontName='Arial',fontSize=7.8,fillColor=GRAY))
    return drawing


def line_chart(points,title,xlabel,ylabel):
    drawing=Drawing(CW,200);x0=47;y0=40;pw=CW-65;ph=125
    drawing.add(String(0,186,title,fontName='Arial-Bold',fontSize=10.5,fillColor=NAVY))
    ymax=max(value for _,value in points)*1.08;xmax=max(key for key,_ in points)
    for i in range(5):
        y=y0+i*ph/4;drawing.add(Line(x0,y,x0+pw,y,strokeColor=colors.HexColor('#dce4eb'),strokeWidth=.5))
        drawing.add(String(x0-7,y-3,f'{ymax*i/4:.2f}',textAnchor='end',fontName='Arial',fontSize=7.5,fillColor=GRAY))
    coordinates=[]
    for x,y in points:
        xx=x0+pw*x/xmax;yy=y0+ph*y/ymax;coordinates.extend([xx,yy])
        drawing.add(Circle(xx,yy,3,fillColor=TEAL,strokeColor=None))
        if x!=1:
            drawing.add(String(xx,y0-15,str(x),textAnchor='middle',fontName='Arial',fontSize=8,fillColor=GRAY))
    drawing.add(PolyLine(coordinates,strokeColor=TEAL,strokeWidth=1.8,fillColor=None))
    drawing.add(String(x0+pw/2,6,xlabel,textAnchor='middle',fontName='Arial',fontSize=8,fillColor=GRAY))
    drawing.add(String(x0,172,ylabel,fontName='Arial',fontSize=8,fillColor=GRAY))
    return drawing


def flow_diagram():
    drawing=Drawing(CW,85);labels=[('Prepare text','Keep raw records'),('Retrieve','Name + address'),('Score pairs','20 similarities'),('Return matches','Threshold + sets')]
    for i,(title,subtitle) in enumerate(labels):
        x=i*(CW/4);width=CW/4-13
        drawing.add(Rect(x,20,width,50,fillColor=LIGHT,strokeColor=None))
        drawing.add(String(x+width/2,50,title,textAnchor='middle',fontName='Arial-Bold',fontSize=9,fillColor=NAVY))
        drawing.add(String(x+width/2,34,subtitle,textAnchor='middle',fontName='Arial',fontSize=7.5,fillColor=GRAY))
        if i<3:drawing.add(String(x+width+3,43,'>',fontName='Arial-Bold',fontSize=12,fillColor=TEAL))
    return drawing


def footer(canvas,doc):
    canvas.saveState();canvas.setFillColor(NAVY);canvas.rect(0,H-8,W,8,fill=1,stroke=0)
    if doc.page>1:
        canvas.setFont('Arial',8);canvas.setFillColor(GRAY)
        canvas.drawString(M,H-28,'ML LAB PROJECT | BUSINESS ENTITY DISAMBIGUITION')
    canvas.setStrokeColor(colors.HexColor('#d8e1e9'));canvas.line(M,34,W-M,34)
    canvas.setFont('Arial',8);canvas.setFillColor(GRAY)
    canvas.drawString(M,22,'Utsav Ojha | Vedant Gajare | Kavya Zala | Sarvesh Zanke')
    canvas.drawRightString(W-M,22,str(doc.page));canvas.restoreState()


manifest=load_json(LAB/'data/dataset_small_10k/manifest.json')
evaluation=load_json(LAB/'artifacts/model_comparison_v1/evaluation/metrics.json')
run=load_json(LAB/'artifacts/model_comparison_v1/run.json')
selection=load_json(LAB/'artifacts/model_comparison_v1/selection.json')
comparisons=read_csv(LAB/'artifacts/model_comparison_v1/validation_comparison.csv')
blocking=read_csv(LAB/'artifacts/blocking_v1/validation_metrics.csv')
cleaning=read_csv(LAB/'artifacts/cleaning_v1/validation_comparison.csv')
walk=load_json(LAB/'reports/model_comparison/walkthrough_data.json')
selected=next(item for item in walk['pipelines'] if item['pipeline_id']==selection['pipeline_id'])
lr=next(item for item in walk['pipelines'] if item['variant']==selection['variant'] and item['model']=='logistic_regression')
case=walk['cases'][0];pair=next(item for item in case['variants'][selection['variant']]['pairs'] if item['is_match'])
countries={}
for item in manifest['strata']:countries[item['country']]=countries.get(item['country'],0)+item['selected']

# COVER
story.append(Spacer(1,42));para('MACHINE LEARNING LAB PROJECT REPORT','small');story.append(Spacer(1,22))
para('Business Entity<br/>Disambiguition','cover')
para('A two-stage approach to noisy business-record matching<br/>with interpretable features and explainable predictions','subtitle')
story.append(Spacer(1,25))
table(['Team member','Roll number'],[
    ['Utsav Ojha','241070079'],['Vedant Gajare','241070080'],
    ['Kavya Zala','241070081'],['Sarvesh Zanke','241070082']],[CW*.64,CW*.36])
story.append(Spacer(1,20));para('Report date: 16 November 2026','small')
callout('<b>Project outcome:</b> character TF-IDF candidate generation followed by XGBoost with conservative placeholder cleanup. '
        'A frozen held-out evaluation achieved macro F0.5 <b>0.98944</b>, link precision <b>99.76%</b>, and link recall <b>97.27%</b>.')
para('This report presents the implemented pipeline, controlled experiments, measured results, and an interactive explanation of how the predictions are formed.','body')

# PAGE 2
section(1,'Introduction')
para('Business identity information often arrives as separate records rather than as one clean, universal identifier. '
     'The same organization may appear with abbreviated legal suffixes, reordered words, typographical changes, different scripts, '
     'and incomplete addresses. Business entity disambiguation, also called entity resolution, determines which of these records refer to the same real-world business.')
para('The project investigates a practical two-stage solution. Candidate generation first reduces the number of record comparisons while retaining likely matches. '
     'A supervised classifier then combines several name and address similarities to decide which candidate links to accept. This separation is widely used in entity-resolution research [1,2].')
para('A second goal is to make the process explainable: a student should be able to inspect the text representation, candidate lists, pair features, learned model parameters, and the final threshold decision.')
h2('Project objectives')
para('1. Construct a reproducible, manageable dataset while keeping complete business-level labels.<br/>'
     '2. Explore data quality and compare candidate-generation strategies.<br/>'
     '3. Compare logistic regression and XGBoost under the same feature protocol.<br/>'
     '4. Measure precision, recall, macro F0.5, and stage-specific failures.<br/>'
     '5. Provide interactive, numerical explanations rather than a score-only demonstration.')
para('2. Problem Statement','h1')
para('Given three business-record sources, use Source 1 as the deduplicated reference set. For every Source 1 record, identify all matching records from Sources 2 and 3. '
     'A reference business may have zero, one, or multiple matching records. The solution must return an empty set when no link is accepted; it must not force a nearest match.')
para('Let T<sub>i</sub> be the true target set, C<sub>i</sub> the retrieved candidates, and M<sub>i</sub> the accepted matches for business i. '
     'The pipeline enforces M<sub>i</sub> as a subset of C<sub>i</sub>. A true match excluded by blocking cannot be recovered by the classifier.')
callout('False merges are especially important: connecting two different businesses is often more damaging than missing one noisy duplicate. '
        'The primary project metric is therefore precision-oriented macro F0.5, alongside link precision and recall.')

# PAGE 3
section(3,'Dataset Description')
para('The supplied data consists of tab-separated source files with four fields: <b>entity_id</b>, <b>business_name</b>, '
     '<b>business_address</b>, and <b>country</b>. Source membership is encoded in record IDs. A separate ground-truth file maps each Source 1 ID to a comma-separated list of target IDs; an empty list is a valid singleton label.')
h2('Reduced, business-level sample')
para('A fixed-seed sample of <b>10,000 reference businesses</b> was selected jointly by country and true-match-count categories '
     '(0, 1, 2, 3, 4, and 5+). All 34,609 labeled target records for these businesses were retained. Another 10,000 target records '
     'were sampled as distractors. Independently sampling rows from each source was avoided because it could remove true matches while keeping their reference records.')
table(['Split','Source 1','Source 2','Source 3','True links'],[
    ['Training','7,000','15,218','16,007','24,225'],['Validation','1,500','3,250','3,439','5,189'],
    ['Evaluation','1,500','3,281','3,414','5,195']],[92,96,100,100,CW-388])
para('Reference businesses comprise <b>5,998 US</b> and <b>4,002 India</b> records. All records belonging to a selected business remain in one split, and record IDs do not overlap between splits. '
     'The TSV files occupy about 5.75 MB, compared with 1.33 GB for the original labeled input files. Sampling used seed 42.')
h2('Training-data observations')
table(['Observation','Measured finding'],[
    ['Missing names','0 across 38,225 training source records'],
    ['Missing / blank addresses','1,247 training records'],
    ['Reference singletons','392 of 7,000 training businesses (5.60%)'],
    ['Name variation','21,696 of 24,225 true pairs differ after lowercase/trim'],
    ['Address variation','21,473 true pairs differ after the same limited comparison'],
    ['Embedded placeholders','1,062 address components; six explicit name markers']],[150,CW-150])
para('The distractor proportion and the smaller target pool are deliberate experimental choices, not estimates of the natural population. '
     'Raw files, sampling settings, memberships, hashes, and verification results were preserved for reproducibility.','small')

# PAGE 4
section(4,'Algorithm Details')
story.append(flow_diagram());para('Figure 1. The implemented two-stage matching pipeline. Only candidate pairs reach the final classifier.','caption')
h2('4.1 Text preparation and controlled cleaning')
para('The baseline creates derived retrieval views using Unicode NFKC, case folding, punctuation/symbol spacing, and whitespace collapsing. '
     'Original strings remain unchanged. Non-Latin characters, accents, and meaningful Unicode non-joiners are retained.')
para('Two conservative alternatives were tested. <b>Placeholder cleanup</b> removes verified missing-value markers from address components and the explicit '
     '&lt;CITY_NAME&gt; marker in names. <b>Cleanup plus abbreviations</b> also expands a small field-specific map such as pvt/private, ltd/limited, and rd/road. '
     'Legitimate names such as "Null Industries" are preserved. Ambiguous St/No tokens, broad typo correction, and transliteration are excluded.')
h2('4.2 Candidate generation: country partitions and TF-IDF')
para('An exact normalized-name blocker provides the baseline. The main blocker builds separate name and address character-ngram TF-IDF views '
     'using fragment lengths 3-5. Vocabulary and inverse-document-frequency weights are fitted on training records only. '
     'Country partitions are discovered dynamically, with explicit fallbacks for blank or absent country labels [1,2].')
para('For fragment t, document frequency df(t), raw count tf(t,d), and N training records:','small')
para('idf(t) = 1 + ln((1 + N) / (1 + df(t)))<br/>'
     'v(t,d) = tf(t,d) x idf(t), followed by L2 normalization<br/>'
     'cosine(q,d) = sum<sub>t</sub> normalized_v(t,q) x normalized_v(t,d)','formula')
para('For each reference business, retrieve the top K positive-cosine records per view and per target source. Take the deduplicated union of the four lists. '
     'K=10 therefore permits up to 40 candidates, not just 10 overall. Blank/zero vectors do not return arbitrary tied records. '
     'Sparse multiplication uses batches of 64 queries rather than a full dense all-pairs matrix.')
callout('The blocker selects alternatives, not final matches. Name and address retrieval complement each other: a weak name can still be recovered through its address, and vice versa.')

# PAGE 5
story.append(PageBreak());para('4. Algorithm Details - continued','h1')
h2('4.3 Pairwise similarity features')
para('Each candidate pair receives the same ten feature types for the business name and address, giving 20 numeric inputs in [0,1]. '
     'IDs, labels, source labels, distractor membership, and country one-hot variables are excluded from the model matrix.')
table(['Feature type','Meaning'],[
    ['TF-IDF cosine','Weighted character-fragment overlap'],['Token Jaccard','Intersection / union of token sets'],
    ['Edit similarity','Normalized unit-cost Levenshtein similarity'],['Length ratio','Shorter string length / longer string length'],
    ['Exact agreement','Equal normalized text, with both fields present'],['Number Jaccard','Overlap of digit-token sets; leading zeros retained'],
    ['Number conflict','Both digit sets exist but differ'],['Missing-left indicator','Whether the reference field is blank'],
    ['Missing-right indicator','Whether the candidate field is blank'],['Token containment','Overlap / smaller nonempty token-set size']],[150,CW-150])
h2('4.4 Logistic regression')
para('Logistic regression standardizes features using training means and scales, then learns one coefficient per feature and an intercept. '
     'L-BFGS minimizes regularized logistic loss, with C=1. Empty fields produce zero similarities plus explicit missingness indicators.')
para('z<sub>j</sub> = (x<sub>j</sub> - mean<sub>j</sub>) / scale<sub>j</sub><br/>'
     'margin = intercept + sum<sub>j</sub> coefficient<sub>j</sub> x z<sub>j</sub><br/>'
     'score = 1 / (1 + exp(-margin))','formula')
tr=lr['training'];values=pair['features'];z=[(x-m)/s for x,m,s in zip(values,tr['mean'],tr['scale'])]
contributions=[a*b for a,b in zip(z,tr['coefficients'])];order=sorted(range(20),key=lambda i:-abs(contributions[i]))[:3]
table(['Worked feature','Value','Coeff.','Std. value','Contribution'],[
    [walk['features'][i].replace('_',' '),f'{values[i]:.4f}',f"{tr['coefficients'][i]:.4f}",f'{z[i]:.4f}',f'{contributions[i]:.4f}'] for i in order],
    [165,73,80,85,CW-403])
para(f'Worked validation pair {case["id"]} / {pair["target_id"]}. The three largest terms are shown; all 20 terms and the intercept contribute to the saved score. '
     'Arithmetic is rounded for display. The walkthrough exposes the complete weight table.','caption')

# PAGE 6
story.append(PageBreak());para('4. Algorithm Details - continued','h1')
h2('4.5 XGBoost and learned leaf weights')
para('XGBoost adds shallow regression trees to a raw score, using gradients and Hessians of binary logistic loss to select splits and leaf updates [3]. '
     'This project uses 200 depth-3 trees, learning rate 0.05, and L2 leaf regularization lambda=1. Training is unweighted and uses all retrieved candidate pairs.')
para('g = current_score - label; h = current_score x (1 - current_score)<br/>'
     'unshrunk_leaf_weight = -sum(g) / (sum(h) + lambda)<br/>'
     'new_margin = previous_margin + learning_rate x leaf_weight','formula')
para('Stored leaf outputs already contain learning-rate shrinkage and are added once. The final sigmoid converts the accumulated margin into a model score; '
     'a validation-selected threshold converts that score into an accepted/rejected link. Scores are not assumed calibrated probabilities.')
tree=selected['training']['trees'][0];node=tree;path=[]
while 'leaf' not in node:
    index=int(node['split'][1:]);value=values[index];go_yes=value<node['split_condition'];next_id=node['yes'] if go_yes else node['no']
    path.append([walk['features'][index].replace('_',' '),f'{value:.5f}',f"{node['split_condition']:.5f}",'Yes' if go_yes else 'No'])
    node=next(child for child in node['children'] if child['nodeid']==next_id)
table(['Actual first-tree split','Pair value','Threshold','Value < threshold?'],path,[185,90,90,CW-365])
stats=selected['training']['leaf_statistics']['0'][str(node['nodeid'])]
para(f'This example reaches leaf {node["nodeid"]}, whose training cohort contains {stats["rows"]:,} pairs. '
     f'Reconstructed sum(g)={stats["gradient_sum"]:.3f}, sum(h)={stats["hessian_sum"]:.3f}; '
     f'the shrunken Newton update is {stats["expected_stored_output"]:.6f}, matching the saved leaf output {node["leaf"]:.6f}.','small')
loss=[(item['round'],item['loss']) for item in selected['training']['training_loss_curve']]
story.append(line_chart(loss,'Training loss reconstructed from frozen tree prefixes','Number of included trees','Training log loss'))
para('Figure 2. Reconstructed training-only losses at saved prefixes; this is not a logged optimizer trace or a validation curve. '
     'The walkthrough also shows actual paths at rounds 1, 25, 100, and 200.','caption')
para('Local explanations use standardized coefficient contributions for logistic regression and native SHAP contributions for XGBoost [4]. '
     'They reconstruct raw log-odds, not causal effects. SHAP uses an expected-margin reference distinct from the initial boosting margin.','small')

# PAGE 7
section(5,'Training Information')
table(['Setting','Logistic regression','XGBoost'],[
    ['Input features','Same fixed 20 features','Same fixed 20 features'],['Preprocessing','Training-only StandardScaler','No standardization required'],
    ['Capacity / regularization','C=1, L2 penalty','200 trees, depth 3, lambda=1'],['Optimizer / tree method','L-BFGS, max_iter=1000','Histogram; min_child_weight=5'],
    ['Learning rate','Solver determined','0.05'],['Sampling / weighting','No negative subsampling or class weighting','subsample=1, colsample_bytree=1'],
    ['Random seed','42','42'],['Parallelism','Numerical threads limited to 2','2 CPU threads']],[136,180,CW-316])
h2('Fit, select, freeze, evaluate')
para('1. Use the saved business-level splits and K=10 candidate sets.<br/>'
     '2. Build the shared features for each of the three text/candidate variants.<br/>'
     '3. Fit both classifiers on the same training pairs within each variant.<br/>'
     '4. Sweep every distinct validation score, processing tied scores together.<br/>'
     '5. Choose each threshold by macro F0.5, then compare the six pipelines.<br/>'
     '6. Freeze the winning model, vectorizers, cleaning rules, feature order, and threshold.<br/>'
     '7. Save evaluation predictions before reading evaluation labels. Do not refit or retune afterward.')
para('Unretrieved positives remain false negatives in end-to-end scoring. No label-driven candidate insertion is performed. '
     'There is no broad hyperparameter search, early stopping on held-out data, or external identity lookup.')
h2('Environment and reproducibility')
versions=run['versions']
para('The project uses Python '+versions['python']+', Polars '+versions['polars']+', scikit-learn '+versions['sklearn']+', XGBoost '+versions['xgboost']+
     ', and RapidFuzz '+versions['rapidfuzz']+'. Dependencies are managed with uv and a lockfile. '
     'The complete model comparison and frozen evaluation took about '+f"{run['elapsed_seconds']:.1f}"+' seconds on the execution machine; timing is an observed run, not a hardware-independent benchmark.')
para('Saved artifacts include input hashes, protocol, feature contract, fitted models, validation scores and threshold curves, '
     'native XGBoost export, error tables, and the frozen evaluation outputs. Tests cover metrics, missingness, tied scores, training-only standardization, TF-IDF calculations, and weight reconstruction.','small')

# PAGE 8
section(6,'Results and Analysis')
h2('6.1 Candidate-generation comparison')
chosen_block=[row for row in blocking if row['method'] in ['exact_name','tfidf_union_k10','tfidf_union_k25','tfidf_union_k50']]
labels=['Exact name','TF-IDF K=10','TF-IDF K=25','TF-IDF K=50']
table(['Method','Retained true pairs','Blocking recall','Candidate pairs'],[
    [label,f"{int(row['retained_true_pairs']):,} / 5,189",pct(row['pair_recall'],3),f"{int(row['candidate_pairs']):,}"]
    for label,row in zip(labels,chosen_block)],[120,139,120,CW-379])
story.append(bar_chart(['Exact\nname','TF-IDF\nK=10','TF-IDF\nK=25','TF-IDF\nK=50'],
    [float(row['pair_recall']) for row in chosen_block],'Validation blocking recall',format_value=lambda value:pct(value,2)))
para('Figure 3. Name/address union retrieval retains far more true links than exact-name blocking. K=10 was selected by the predefined minimum-cost rule among settings meeting 98% recall.','caption')
para('At K=10, blocking retained 5,182 of 5,189 validation true links and reduced 10,033,500 possible comparisons to 55,430 candidates, '
     'a 99.45% reduction. K=25 and K=50 retained all observed validation true links but increased candidate cost. Finite observed recall is not a guarantee on every future record.')
h2('6.2 Cleaning ablation at fixed K=10')
table(['Retrieval text','Recall','Candidates','Recovered / newly lost'],[
    ['Baseline','99.865%','55,430','0 / 0'],['Placeholder cleanup','99.865%','55,426','0 / 0'],
    ['Cleanup + abbreviations','99.865%','55,425','0 / 0']],[153,88,100,CW-341])
para('Cleaning did not change validation true-pair coverage. It did change candidate identities: placeholder cleanup added 319 and removed 323 candidates; '
     'abbreviation expansion added 4,871 and removed 4,876. A tiny net count change therefore does not imply unchanged classifier inputs.','small')

# PAGE 9
story.append(PageBreak());para('6. Results and Analysis - continued','h1')
h2('6.3 Model comparison on validation')
variant_names={'baseline':'Baseline','placeholders':'Placeholder cleanup','placeholders_abbreviations':'Cleanup + abbreviations'}
table(['Text / candidates','Model','Macro F0.5','Precision','Recall'],[
    [variant_names[row['variant']],'LR' if row['model']=='logistic_regression' else 'XGBoost',
     f"{float(row['macro_f05']):.6f}",f"{float(row['micro_precision']):.6f}",f"{float(row['micro_recall']):.6f}"] for row in comparisons],
    [153,76,92,92,CW-413])
story.append(bar_chart(['Baseline\nLR','Baseline\nXGBoost','Cleanup\nLR','Cleanup\nXGBoost','Expanded\nLR','Expanded\nXGBoost'],
    [float(row['macro_f05']) for row in comparisons],'Validation macro F0.5 (zoomed axis: 0.97-1.00)',low=.97,high=1,
    format_value=lambda value:f'{value:.5f}',color_list=[BLUE,TEAL,BLUE,TEAL,BLUE,TEAL]))
para('Figure 4. Fixed model comparisons with a deliberately zoomed vertical axis; exact values are in the table. LR denotes logistic regression.','caption')
para('XGBoost performed better than logistic regression for each text/candidate variant in this first fixed comparison. '
     '<b>XGBoost with placeholder cleanup</b> obtained the highest validation macro F0.5 and was selected at threshold <b>0.4353279173</b>.')
para('Its advantage over baseline XGBoost is only 0.000094235 in macro F0.5, approximately 0.0094 percentage points. '
     'This is a narrow validation win, not strong evidence of a general cleaning benefit. Abbreviation expansion lowered validation scores for both models in this comparison.')
callout('Cleaning affects both retrieval and pair features. These results compare complete pipelines, not isolated causal effects of one text-cleaning rule.')

# PAGE 10
story.append(PageBreak());para('6. Results and Analysis - continued','h1')
h2('6.4 Frozen held-out evaluation')
table(['Metric','Result'],[
    ['Macro F0.5',f"{evaluation['macro_f05']:.6f}"],['Link precision',pct(evaluation['micro_precision'],3)],
    ['Link recall',pct(evaluation['micro_recall'],3)],['Link F1',f"{evaluation['micro_f1']:.6f}"],
    ['Exact-set accuracy',pct(evaluation['exact_set_accuracy'])],['Correct singleton outputs','83 / 83'],
    ['True positives / false positives / false negatives','5,053 / 12 / 142']],[239,CW-239])
ci=evaluation['macro_f05_bootstrap_95']
para(f"The evaluation macro F0.5 was {evaluation['macro_f05']:.6f}, with a 95% bootstrap interval [{ci['low']:.6f}, {ci['high']:.6f}]. "
     'The interval uses 2,000 resamples of reference businesses with seed 42. It conditions on the fixed selected pipeline and excludes training/selection uncertainty.')
story.append(bar_chart(['Lost in\nblocking','Rejected by\nclassifier','False\nmerges'],[12,130,12],
    'Evaluation error decomposition',high=150,format_value=lambda value:str(int(value)),color_list=[BLUE,TEAL,RED]))
para('Figure 5. The 142 missed true links comprise 12 blocking misses and 130 classifier rejections. The 12 false merges are a separate error category.','caption')
h2('6.5 Metric interpretation')
para('For each reference business, TP, FP, and FN are set-based link counts. Its F0.5 is computed before averaging across businesses:','small')
para('F0.5<sub>i</sub> = 1.25 TP<sub>i</sub> / (1.25 TP<sub>i</sub> + FP<sub>i</sub> + 0.25 FN<sub>i</sub>)','formula')
para('When both true and predicted sets are empty, the per-business score is 1. Other zero-overlap cases score 0. '
     'Link precision and recall pool counts across businesses, whereas macro F0.5 gives each reference business equal weight. '
     'Exact-set accuracy is stricter: a single missing or extra link makes that business incorrect. Precision-recall analysis is useful for this highly imbalanced candidate task [5].','small')

# PAGE 11
section(7,'Conclusion')
para('The project demonstrates an efficient and explainable approach to business entity disambiguation. '
     'Character-ngram TF-IDF retrieval retained most true matches while sharply reducing comparison cost. '
     'A shared set of 20 similarity features supported a fair first comparison of logistic regression and XGBoost across three preprocessing variants.')
para('XGBoost with conservative placeholder cleanup was selected on validation and achieved held-out macro F0.5 of 0.98944, '
     'link precision of 99.76%, and link recall of 97.27%. The explanation interface connects each result to actual retrieval weights, '
     'candidate ranks, feature values, learned coefficients or tree leaf updates, and the decision threshold.')
h2('Limitations')
para('<b>Reduced-pool scope.</b> The constructed target pool and added distractors influence task difficulty. '
     'The measured performance should be interpreted for this sampling design, rather than as a population-wide guarantee.')
para('<b>Lexical coverage.</b> Missing addresses, cross-script names, abbreviations, and severe text corruption can defeat character overlap. '
     'The initial blocker imposes a recall ceiling, and the model cannot recover a pair it never receives.')
para('<b>Threshold and score interpretation.</b> Each decision threshold was chosen on validation; scores were not calibrated as real-world match probabilities. '
     'Changes in prevalence or input quality may require renewed validation.')
para('<b>Finite evidence.</b> The cleaning advantage was small, and the model comparison used a bounded parameter configuration. '
     'Observed zero singleton false merges among 83 evaluation cases is not a zero-risk guarantee. The bootstrap interval does not include model-selection uncertainty.')
para('<b>Evaluation discipline.</b> The held-out split has now been used for one frozen assessment. Future algorithm selection should stay on training/validation '
     'and use a fresh holdout when an independent final estimate is needed.')
h2('Future project extensions')
para('Investigate missing-field and cross-script failures using training diagnostics; compare additional retrieval representations and classifiers under a frozen protocol; '
     'test probability calibration and cost-sensitive thresholds; and evaluate scaling with larger locally available target pools. '
     'Any proposed enhancement should report recovered and newly lost links as well as aggregate scores.')

# PAGE 12
section(8,'References')
refs=[
('[1]','G. Papadakis, D. Skoutas, E. Thanos, and T. Palpanas. <i>A Survey of Blocking and Filtering Techniques for Entity Resolution.</i> arXiv:1905.06167, revised 2020.','https://arxiv.org/abs/1905.06167'),
('[2]','G. Papadakis, M. Fisichella, F. Schoger, G. Mandilaras, N. Augsten, and W. Nejdl. <i>How to reduce the search space of Entity Resolution: with Blocking or Nearest Neighbor search?</i> arXiv:2202.12521, 2022.','https://arxiv.org/abs/2202.12521'),
('[3]','T. Chen and C. Guestrin. <i>XGBoost: A Scalable Tree Boosting System.</i> Proceedings of the 22nd ACM SIGKDD International Conference, 2016. DOI: 10.1145/2939672.2939785.','https://arxiv.org/abs/1603.02754'),
('[4]','S. M. Lundberg, G. G. Erion, and S.-I. Lee. <i>Consistent Individualized Feature Attribution for Tree Ensembles.</i> arXiv:1802.03888, 2018; revised 2019.','https://arxiv.org/abs/1802.03888'),
('[5]','T. Saito and M. Rehmsmeier. <i>The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.</i> PLOS ONE 10(3), e0118432, 2015.','https://doi.org/10.1371/journal.pone.0118432')]
for index,text,url in refs:
    para(f'<b>{index}</b> {text}<br/><link href="{url}" color="#087f8c">Open paper: {escape(url)}</link>')
h2('Implementation documentation')
for label,url in [
    ('scikit-learn: logistic regression','https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'),
    ('scikit-learn: smoothed TF-IDF','https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html'),
    ('XGBoost: boosted-tree weight derivation','https://xgboost.readthedocs.io/en/stable/tutorials/model.html'),
    ('XGBoost: prediction and feature contributions','https://xgboost.readthedocs.io/en/stable/prediction.html'),
    ('RapidFuzz: normalized Levenshtein similarity','https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html')]:
    para(f'<link href="{url}" color="#087f8c">{escape(label)}</link>','small')
para('References motivate the methods; all project performance numbers are measured from the saved local experiments. '
     'Paper hyperlinks open the original source pages, where full-text PDFs are available. Web references accessed 16 November 2026.','small')

path=OUT/'Business_Entity_Disambiguition_Report.pdf'
document=SimpleDocTemplate(str(path),pagesize=A4,leftMargin=M,rightMargin=M,topMargin=48,bottomMargin=48,
    title='Business Entity Disambiguition',author='Utsav Ojha; Vedant Gajare; Kavya Zala; Sarvesh Zanke',
    subject='ML lab project: candidate generation, pair features, logistic regression, XGBoost and explainability')
document.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(path)
texts=[page.extract_text() for page in reader.pages]
assert all(len(text)>120 for text in texts)
full='\n'.join(texts)
for token in ['Utsav Ojha','241070079','Vedant Gajare','241070080','Kavya Zala','241070081','Sarvesh Zanke','241070082',
              'Introduction','Problem Statement','Dataset Description','Algorithm Details','Training Information','Results and Analysis','Conclusion','References']:
    assert token in full,token
assert 'competition' not in full.lower()
links=sum(len(page.get('/Annots',[])) for page in reader.pages)
assert links>=10
(OUT/'report_validation.json').write_text(json.dumps({'pages':len(reader.pages),'paper_and_documentation_links':links,
    'required_sections_and_team_details_verified':True,'project_focused_limitations':True,'source':'saved local artifacts; no fitting or evaluation rerun'},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'pdf':str(path),'pages':len(reader.pages),'bytes':path.stat().st_size,'links':links},indent=2))
