// The Confidence writeup's example, chart and code tabs. The example values are illustrative, as the page says.
(function(){
const $=id=>document.getElementById(id);
function el(t,c,x){const e=document.createElement(t);if(c)e.className=c;if(x!=null)e.textContent=x;return e}

const T=[
 {text:"I ordered two lamps, #4471 and #4472, and neither has arrived. Where are they?",
  f:[{k:"category",kind:"enum",v:"shipping",c:0.94,d:{billing:0.03,technical:0.01,shipping:0.94,other:0.02}},
     {k:"urgent",kind:"boolean",v:"false",c:0.88,d:{"true":0.12,"false":0.88}},
     {k:"orders",kind:"integer",v:"2",c:0.97}]},
 {text:"I was charged twice for order #123. - Ann",
  f:[{k:"category",kind:"enum",v:"billing",c:0.91,d:{billing:0.91,technical:0.02,shipping:0.03,other:0.04}},
     {k:"urgent",kind:"boolean",v:"false",c:0.79,d:{"true":0.21,"false":0.79}},
     {k:"orders",kind:"integer",v:"1",c:0.99}]},
 {text:"The tracking link you sent for #9020 gives an error page, and I need it before we move out on Friday.",
  f:[{k:"category",kind:"enum",v:"shipping",c:0.60,d:{billing:0.02,technical:0.37,shipping:0.60,other:0.01}},
     {k:"urgent",kind:"boolean",v:"true",c:0.56,d:{"true":0.56,"false":0.44}},
     {k:"orders",kind:"integer",v:"1",c:0.98}]}
];
let ti=0;
const KIND={enum:"choice from a list, reported as the chosen option's share",boolean:"yes or no",integer:"a count, reported as its least certain digit"};
function render(){
  const t=T[ti], th=+$("th").value; $("tt").textContent=t.text; $("tho").textContent=th.toFixed(2);
  const ul=$("flds"); ul.textContent="";
  t.f.forEach(f=>{
    const li=el("li","fld"); const a=el("div");
    a.appendChild(el("div","k",f.k));
    const v=el("div","v"); v.appendChild(el("span",null,f.v)); v.appendChild(el("span","c "+(f.c<th?"lo":"hi"),f.c.toFixed(2))); a.appendChild(v);
    a.appendChild(el("div","kind",KIND[f.kind]));
    const b=el("div");
    if(f.d){ const d=el("div","dist");
      Object.entries(f.d).sort((x,y)=>y[1]-x[1]).forEach(([o,p])=>{const pick=o===f.v;
        d.appendChild(el("span","o"+(pick?" pick":""),o)); const tr=el("div","track"); const fl=el("div","f"+(pick?" pick":"")); fl.style.width=(p*100)+"%"; tr.appendChild(fl);
        if(pick){const m=el("span","th");m.style.left="calc("+(th*100)+"% - 1px)";m.title="threshold";tr.appendChild(m)}
        d.appendChild(tr); d.appendChild(el("span","p",p.toFixed(2)));});
      b.appendChild(d);
    } else b.appendChild(el("p","intnote","One digit, and the model gave it "+Math.round(f.c*100)+"% of what the schema allowed. Integers carry no distribution: there's no closed list of options to spread it over."));
    li.appendChild(a); li.appendChild(b); ul.appendChild(li);
  });
  const low=t.f.filter(f=>f.c<th); const v=$("verdict"); v.textContent="";
  if(low.length){v.className="verdict person";v.appendChild(document.createTextNode("Send to a person"));v.appendChild(el("small",null,low.map(f=>f.k+" "+f.c.toFixed(2)).join(", ")+" below "+th.toFixed(2)));}
  else {v.className="verdict auto";v.appendChild(document.createTextNode("Handle automatically"));v.appendChild(el("small",null,"every field at or above "+th.toFixed(2)));}
}
$("tix").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{ti=+b.dataset.i;$("tix").querySelectorAll("button").forEach(x=>x.setAttribute("aria-pressed",String(x===b)));render()}));
$("th").addEventListener("input",render);
$("codetabs").querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{$("codetabs").querySelectorAll("button").forEach(x=>x.setAttribute("aria-pressed",String(x===b)));$("c-http").hidden=b.dataset.c!=="http";$("c-go").hidden=b.dataset.c!=="go"}));

/* AUROC scale: 0.5 .. 1.0 */
(function(){
  const s=$("scale"), X=v=>((v-0.5)/0.5*100)+"%";
  const band=el("div","band"); band.style.left="0"; band.style.width=X(0.55); band.title="below 0.55: fails"; s.appendChild(band);
  const g=el("div","gate"); g.style.left=X(0.65); g.appendChild(el("span",null,"bar set in advance: 0.65")); s.appendChild(g);
  s.appendChild(el("div","axis"));
  [0.5,0.6,0.7,0.8,0.9,1.0].forEach(v=>{const t=el("span","tick",v.toFixed(1));t.style.left=X(v);s.appendChild(t)});
  const P=[["enum",0.847,0,10],["boolean",0.727,0,35],["integer",0.680,0,10],["enum",0.967,1,60],["boolean",0.905,1,35],["integer",0.942,1,10]];
  P.forEach(([k,v,h,y])=>{const d=el("span","pt"+(h?" hollow":""));d.style.left=X(v);d.style.top=(y+14)+"px";d.title=k+" "+v;s.appendChild(d);
    const l=el("span","pl"+(h?" h":""),k+" "+v.toFixed(2));l.style.left=X(v);l.style.top=(y-8)+"px";s.appendChild(l)});
})();
render();
})();
