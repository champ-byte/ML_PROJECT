// Functional control/arithmetic checks using a DOM stub; not browser layout QA.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const root=path.resolve(__dirname,'..'),folder=path.join(root,'reports/model_comparison');
const data=JSON.parse(fs.readFileSync(path.join(folder,'side_by_side_data.json'),'utf8'));
const template=fs.readFileSync(path.join(__dirname,'side_by_side.html'),'utf8');
const app=template.slice(template.indexOf("'use strict';"),template.lastIndexOf('</script>'));
class Element{
 constructor(){this.value='';this.textContent='';this.innerHTML='';this.listeners={};this.attributes={};this.hidden=false;}
 replaceChildren(...items){this.children=items;this.value=items[0]?.value??'';}
 addEventListener(event,fn){this.listeners[event]=fn;}
 setAttribute(key,value){this.attributes[key]=value;}
}
const elements=new Map([...template.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Element()]));
const E=id=>elements.get(id),plots=new Map();E('payload').textContent=JSON.stringify(data);
E('coefficientBasis').value='standardized';
const context=vm.createContext({document:{getElementById:E},Option:function(text,value){this.text=text;this.value=value;},
 Plotly:{react:(id,traces,layout)=>{assert(elements.has(id));for(const trace of traces)assert(trace.x.every(Number.isFinite));plots.set(id,{traces,layout});}}});
vm.runInContext(app,context);
let scored=0,missing=0,disagreements=0;
for(const c of data.cases){
 E('business').value=c.id;E('business').listeners.change();
 const union=new Set(data.variants.flatMap(v=>c.variants[v].pairs.map(p=>p.target_id)));
 for(const variant of data.variants){
  E('variant').value=variant;E('variant').listeners.change();
  for(const id of union){
   E('candidate').value=id;E('candidate').listeners.change();
   const pair=c.variants[variant].pairs.find(p=>p.target_id===id),models=data.models.filter(p=>p.variant===variant);
   for(const p of models){
    const prefix=p.model==='xgboost'?'xgb':'lr',summary=E(prefix+'Summary').innerHTML;
    const chart=plots.get(prefix+'Contributions');
    if(!pair){assert(summary.includes('Not retrieved / not scored'));assert.equal(chart.traces.length,0);continue;}
    assert(summary.includes(pair.scores[p.model].toFixed(6)));
    assert(summary.includes(p.threshold.toFixed(6)));
    assert(summary.includes(pair.scores[p.model]>=p.threshold?'ACCEPT MATCH':'REJECT MATCH'));
    assert.deepEqual(Array.from(chart.traces[0].x),pair.contributions[p.model].slice(0,-1));
    const margin=pair.contributions[p.model].reduce((a,b)=>a+b,0);
    assert(Math.abs(1/(1+Math.exp(-margin))-pair.scores[p.model])<3e-6);
    if(p.model==='logistic_regression'){
     const reconstructed=p.training.intercept+pair.features.reduce((sum,value,i)=>sum+(value-p.training.mean[i])/p.training.scale[i]*p.training.coefficients[i],0);
     // Training standardization used float32; display arithmetic uses JS doubles.
     assert(Math.abs(1/(1+Math.exp(-reconstructed))-pair.scores[p.model])<2e-6);
    }
   }
   assert.deepEqual(Array.from(plots.get('lrContributions').layout.xaxis.range),Array.from(plots.get('xgbContributions').layout.xaxis.range));
   if(pair){scored++;const [a,b]=models.map(p=>pair.scores[p.model]>=p.threshold);if(a!==b){disagreements++;assert(E('decisionDifference').innerHTML.includes('disagree'));}}
   else{missing++;assert(E('featureTable').innerHTML.includes('unavailable'));}
   for(const other of data.variants){E('variant').value=other;E('variant').listeners.change();assert.equal(E('candidate').value,id);}
   E('variant').value=variant;E('variant').listeners.change();
  }
 }
}
E('weightsTab').listeners.click();assert.equal(E('weightsPanel').hidden,false);
for(const variant of data.variants){
 E('variant').value=variant;E('variant').listeners.change();
 for(const basis of ['standardized','raw']){
  E('coefficientBasis').value=basis;E('coefficientBasis').listeners.change();
  const traces=plots.get('coefficientPlot').traces;
  for(const [i,v] of data.variants.entries()){
   const p=data.models.find(p=>p.variant===v&&p.model==='logistic_regression');
   const expected=p.training.coefficients.map((b,j)=>basis==='raw'?b/p.training.scale[j]:b);
   assert.deepEqual(Array.from(traces[i].x),expected);
  }
 }
 const global=plots.get('globalPlot');assert.equal(global.traces.length,2);
 ['logistic_regression','xgboost'].forEach((name,i)=>assert.deepEqual(Array.from(global.traces[i].x),data.models.find(p=>p.variant===variant&&p.model===name).mean_absolute_contribution));
 assert(E('cohort').textContent.includes(data.global_cohort.pairs.toLocaleString()));
}
E('pairTab').listeners.click();assert.equal(E('pairPanel').hidden,false);
assert(scored>0&&missing>0&&disagreements>0,'Fixture must exercise scored, absent, and disagreement cases');
assert.equal(data.provenance.models_retrained,false);assert.equal(data.provenance.additional_evaluation_inference_runs,0);
for(const page of ['model_comparison.html','interactive_walkthrough.html'])assert(fs.readFileSync(path.join(folder,page),'utf8').includes('href="side_by_side_comparison.html"'));
const html=fs.readFileSync(path.join(folder,'side_by_side_comparison.html'),'utf8');
assert(!/<script[^>]+src=/i.test(html),'Report must work offline');assert(!html.includes('__DATA__'));
console.log(`PASS: ${scored} scored and ${missing} absent same-pair states; ${disagreements} disagreements; variant identity preservation, coefficient units, equal chart scales, contribution arithmetic, global cohorts and offline links. Browser layout not tested.`);
