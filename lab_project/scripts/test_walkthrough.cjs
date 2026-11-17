// Pure JavaScript/DOM-stub checks, not a browser rendering test.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const root=path.resolve(__dirname,'..');
const data=JSON.parse(fs.readFileSync(path.join(root,'reports/model_comparison/walkthrough_data.json'),'utf8'));
const template=fs.readFileSync(path.join(__dirname,'walkthrough.html'),'utf8');
const app=template.slice(template.indexOf("'use strict';"),template.lastIndexOf('</script>'))
 .replace('__LEARNING__',fs.readFileSync(path.join(__dirname,'walkthrough_learning.js'),'utf8'));
const elements=new Map();
class Element{
 constructor(id){this.id=id;this.value='';this.textContent='';this.listeners={};this.attributes={};this.children=[];this.dataset={};this.hidden=false;this._html='';}
 replaceChildren(...values){this.children=values;if(values.length)this.value=values[0].value;}
 setAttribute(key,value){this.attributes[key]=value;}
 addEventListener(key,fn){this.listeners[key]=fn;}
 set innerHTML(text){this._html=text;this.children=[];for(const match of text.matchAll(/data-step="(\d+)"/g)){const e=new Element('step');e.dataset.step=match[1];this.children.push(e);}if(this.id==='stage'){elements.delete('pairSelect');elements.delete('featurePlot');elements.delete('contributionPlot');for(const match of text.matchAll(/id="([^"]+)"/g))elements.set(match[1],new Element(match[1]));}}
 get innerHTML(){return this._html;}
 querySelectorAll(selector){if(selector==='[data-step]')return this.children;if(selector==='[data-target]')return [...this._html.matchAll(/data-target="([^"]+)"/g)].map(match=>{const e=new Element('target');e.dataset.target=match[1];return e;});return [];}
}
for(const match of template.matchAll(/id="([^"]+)"/g))elements.set(match[1],new Element(match[1]));
elements.get('payload').textContent=JSON.stringify(data);
const charts=[];
const context=vm.createContext({document:{getElementById:id=>elements.get(id)||null},
 Option:function(text,value){this.text=text;this.value=value;},
 Plotly:{react:(id,traces,layout)=>{assert(elements.has(id),'Missing chart container '+id);charts.push(id);}},console});
vm.runInContext(app,context);
const E=id=>elements.get(id);
const evaluation=E('evalMacro').textContent;
assert.equal(E('pipeline').value,data.selection.pipeline_id);
let checks=0;
for(const pipeline of data.pipelines){
 E('pipeline').value=pipeline.pipeline_id;E('pipeline').listeners.change();
 assert.equal(E('macro').textContent,(pipeline.metrics.macro_f05*100).toFixed(2)+'%');
 const chosenIndex=+E('threshold').value;
 for(const index of [0,chosenIndex,pipeline.curve.length-1]){
  E('threshold').value=index;E('threshold').listeners.input();
  assert.equal(E('precision').textContent,(pipeline.curve[index].micro_precision*100).toFixed(2)+'%');
  assert.equal(E('evalMacro').textContent,evaluation,'Evaluation changed with validation slider');checks++;
 }
 E('resetThreshold').listeners.click();assert.equal(+E('threshold').value,chosenIndex);
 for(const example of data.cases){
  E('business').value=example.id;E('business').listeners.change();
  for(let step=0;step<5;step++){
   vm.runInContext(`step=${step};render()`,context);
   assert(E('stage').innerHTML.length>40);checks++;
   if(step===1){for(const view of ['name','address']){E('retrievalView').value=view;E('retrievalView').listeners.change();assert(E('stage').innerHTML.includes('Full cosine:'));checks++;}}
   if(step===3&&pipeline.model==='xgboost'){for(const round of pipeline.training.checkpoints){E('boostRound').value=round;E('boostRound').listeners.change();assert(E('stage').innerHTML.includes('Saved leaf output'));checks++;}}
  }
 }
 E('compareTab').listeners.click();E('histFeature').listeners.change();
 assert.equal(E('comparePanel').hidden,false);E('walkTab').listeners.click();
 assert.equal(E('walkPanel').hidden,false);
}
for(const c of data.cases)for(const value of Object.values(c.variants))for(const pair of value.pairs){
 assert.equal(pair.features.length,data.features.length);
 assert(pair.features.every(x=>Number.isFinite(x)&&x>=0&&x<=1.000001));
 for(const name of ['logistic_regression','xgboost']){
  const margin=pair.contributions[name].reduce((a,b)=>a+b,0),pred=1/(1+Math.exp(-margin));
  assert(Math.abs(pred-pair.scores[name])<2e-6,'Contribution/score mismatch');
 }
 for(const [view,index] of [['name',0],['address',10]]){
  const w=pair.tfidf[view];assert(Math.abs(w.cosine-pair.features[index])<2e-6);
  assert(Math.abs(w.shown_product_sum+w.other_product_sum-w.cosine)<1e-12);
 }
}
for(const c of data.cases)for(const value of Object.values(c.variants)){
 const union=new Set();
 for(const source of ['source2','source3'])for(const view of ['name','address']){
  const list=value.bucketing.lists[source][view];assert(list.length<=10);
  list.forEach((row,i)=>{assert.equal(row.rank,i+1);assert(row.cosine>0);union.add(row.target_id);});
 }
 assert.deepEqual([...union].sort(),value.pairs.map(r=>r.target_id).sort(),'Bucketing union differs from scored candidate set');
}
for(const pipeline of data.pipelines){
 for(const c of data.cases)for(const pair of c.variants[pipeline.variant].pairs){
  const t=pipeline.training;
  const reconstructed=vm.runInContext(`(function(){const p=D.pipelines.find(p=>p.pipeline_id===${JSON.stringify(pipeline.pipeline_id)}),c=D.cases.find(c=>c.id===${JSON.stringify(c.id)}),r=c.variants[p.variant].pairs.find(r=>r.target_id===${JSON.stringify(pair.target_id)});if(p.model==='xgboost')return treeProgress(p.training,r.features).scores.at(-1);const z=r.features.map((x,i)=>(x-p.training.mean[i])/p.training.scale[i]);return 1/(1+Math.exp(-(p.training.intercept+z.reduce((s,x,i)=>s+x*p.training.coefficients[i],0))));})()`,context);
  assert(Math.abs(reconstructed-pair.scores[pipeline.model])<3e-6,'Learned model weights do not reconstruct score');
 }
}
assert(charts.includes('featurePlot')&&charts.includes('contributionPlot')&&charts.includes('thresholdPlot')&&charts.includes('weightsPlot')&&charts.includes('growthPlot'));
console.log(`PASS: ${checks} step/threshold state checks across all pipelines/examples; exact score thresholds, fixed evaluation readout, finite features and additive explanations. Browser layout was not tested.`);
