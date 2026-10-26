export class ECGEngine{
  constructor(){this.t=0;this.anom=false}
  next(){this.t+=0.04;const p=this.t%(2*Math.PI),g=(x,m,v)=>Math.exp(-Math.pow(x-m,2)/v);
    const base=0.15*g(p,1,0.02)+1.2*g(p,2,0.005)-0.2*g(p,1.9,0.005)+0.3*g(p,3.2,0.08);
    let o=base+(Math.random()-0.5)*0.03;if(this.anom&&p>2.1&&p<3.0)o+=0.45;
    return{o,b:base,e:Math.abs(o-base)}}
  trigger(ms=4000){this.anom=true;setTimeout(()=>this.anom=false,ms)}
}
