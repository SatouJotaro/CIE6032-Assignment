/* Development-only browser regression harness; writes a temporary standalone page. */
const fs=require('node:fs'),path=require('node:path');
const code=String.raw`
(async()=>{
 const reports=[];
 const assert=(condition,name)=>{if(!condition)throw Error(name);reports.push(name);};
 const tick=()=>new Promise(resolve=>setTimeout(resolve,30));
 const go=async hash=>{location.hash=hash;await tick();};
 try{
  assert(COURSE.length===15,'15 lessons');
  assert(allQuestions.length===45,'45 unique questions');
  assert(new Set(allQuestions.map(q=>q.id)).size===45,'question IDs unique');
  for(const view of ['home','roadmap','labs','review','resources','notes','diagnostic','depth/03','coverage','plan','evidence','search/'+encodeURIComponent('梯度')]){
   await go(view);assert(document.querySelector('main h1'),'route '+view);assert(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow '+view);
  }
  for(const l of COURSE){await go('lesson/'+l.id);assert(document.querySelectorAll('.concept').length===4,'lesson '+l.id+' concepts');assert(document.querySelectorAll('.quiz').length===3,'lesson '+l.id+' questions');assert(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow lesson '+l.id);}
  for(const l of COURSE){await go('depth/'+l.id);assert(document.querySelectorAll('.depth-panel details[open]').length===2,'derivation and code expanded '+l.id);assert(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow depth '+l.id);}
  assert(Object.keys(LABS).length===17,'17 interactive labs');
  assert(EXPERIMENT.runs.length===15,'15 actual experiment runs embedded');
  assert(EXPERIMENT.resume_check.next_update_max_abs_error<1e-7,'checkpoint consistency');
  assert(nmsToy(1).keep.length===4,'NMS threshold 1 retains all candidates');
  assert(nmsToy(0).keep.length<nmsToy(1).keep.length,'NMS lower threshold suppresses candidates');
  for(const kind of Object.keys(LABS)){
   await go('lab/'+kind);const lab=document.querySelector('[data-lab]');assert(lab.querySelector('.lab-visual').innerHTML.length>10,'lab '+kind+' renders');assert(document.documentElement.scrollWidth<=innerWidth,'no horizontal overflow lab '+kind);const control=lab.querySelector('[data-param]'),before=lab.querySelector('.lab-result').textContent;
   if(control.tagName==='SELECT')control.selectedIndex=1;else control.value=control.max;
   control.dispatchEvent(new Event('input',{bubbles:true}));assert(lab.querySelector('.lab-result').textContent!==before,'lab '+kind+' updates');
  }
  await go('lab/gradient');document.querySelector('[data-param=eta]').value='1.1';document.querySelector('[data-param=eta]').dispatchEvent(new Event('input',{bubbles:true}));assert(document.querySelector('.lab-result').textContent.includes('发散'),'gradient divergence');
  await go('lab/iou');for(const [k,v]of[['x',60],['y',45]]){const el=document.querySelector('[data-param='+k+']');el.value=v;el.dispatchEvent(new Event('input',{bubbles:true}));}assert(document.querySelector('.lab-result').textContent.includes('IoU=1.0000'),'IoU identical boxes');
  await go('lesson/01');
  const submit=(qid,choice)=>{const form=document.querySelector('[data-question="'+qid+'"]');form.querySelector('input[value="'+choice+'"]').checked=true;form.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));};
  submit('01a',1);assert(state.answers['01a'].correct===false&&state.mistakes['01a'],'wrong answer tracked');
  submit('01a',0);submit('01b',1);submit('01c',2);
  assert(document.querySelector('#complete-lesson').disabled,'completion blocked without self-checks');
  for(const c of document.querySelectorAll('[data-check]')){c.checked=true;c.dispatchEvent(new Event('change',{bubbles:true}));}
  assert(!document.querySelector('#complete-lesson').disabled,'completion enabled after quiz and checks');
  document.querySelector('#complete-lesson').click();assert(state.completed['01'],'completion recorded');
  const note=document.querySelector('[data-note]');note.value='我的理解 <script>不应执行</script>';note.dispatchEvent(new Event('input',{bubbles:true}));
  assert(JSON.parse(localStorage.getItem(KEY)).notes['01']===note.value,'notes persisted');
  await go('notes');assert(document.querySelector('textarea').value.includes('<script>'),'note text safely restored');
  assert(!document.querySelector('main script'),'notes do not inject script');
  const sanitized=validState(JSON.parse(localStorage.getItem(KEY)));assert(sanitized.completed['01']&&sanitized.mistakes['01a'],'export/import state validation');
  await go('review');assert(wrongQuestions().length===0,'corrected question no longer current mistake');reviewAll=true;route();assert(document.querySelector('[data-question="01a"]'),'historical mistake retained');
  await go('home');assert(document.documentElement.scrollWidth<=innerWidth,'desktop no horizontal overflow');
  localStorage.removeItem(KEY);
  const output=document.createElement('pre');output.id='qa-result';output.textContent=JSON.stringify({pass:true,width:innerWidth,checks:reports.length,reports});document.body.append(output);if(parent!==window)parent.postMessage(output.textContent,'*');
 }catch(e){const output=document.createElement('pre');output.id='qa-result';output.textContent=JSON.stringify({pass:false,width:innerWidth,error:e.message,stack:e.stack,reports});document.body.append(output);if(parent!==window)parent.postMessage(output.textContent,'*');}
})();`;
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
fs.writeFileSync(path.join(__dirname,'qa.html'),html.replace('</body>','<script>'+code.replace(/<\/script/gi,'<\\/script')+'</script></body>'));
console.log('QA page prepared.');
// Chrome CLI has a platform-dependent minimum window layout width. An iframe
// provides an exact 390 CSS-pixel viewport for mobile layout regression checks.
function wrap(content,output){
 const safe=JSON.stringify(content).replace(/<\/script/gi,'<\\/script');
 fs.writeFileSync(path.join(__dirname,output),'<!doctype html><meta charset="utf-8"><style>html,body{margin:0;padding:0}iframe{width:390px;height:844px;border:0;display:block}</style><iframe title="Mobile viewport"></iframe><script>document.querySelector("iframe").srcdoc='+safe+';window.addEventListener("message",e=>{const p=document.createElement("pre");p.id="qa-result";p.textContent=e.data;document.body.append(p)})<\/script>');
}
wrap(fs.readFileSync(path.join(__dirname,'qa.html'),'utf8'),'qa-mobile.html');
wrap(html.replace('window.addEventListener(\'hashchange\',route);route();',"window.addEventListener('hashchange',route);location.hash='lab/attention';route();"),'preview-mobile.html');
