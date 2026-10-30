/* shared per-symbol spread statistics (time-weighted over RTH) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#define NB 400          /* log histogram bins of relative half spread in bp: 10^(-2 + i/50) */
#define NBK 78          /* 5-minute buckets 09:30-16:00 */
#define RTH0 (34200LL*1000000000LL)
#define RTH1 (57600LL*1000000000LL)
typedef struct {
  double tw_t, tw_hs, tw_dsp, tw_mid, invalid_t, locked_t;
  double hist[NB];
  double bk_t[NBK], bk_hs[NBK];
  long nq;
  /* trades */
  double tr_n, tr_dv, tr_eff_dv, tr_eff_n, tr_lit_dv, tr_lit_eff_dv, tr_sh;
  double tr_hist[NB];
} Stats;
static void st_reset(Stats *s){ memset(s,0,sizeof(*s)); }
static int hbin(double hs_bp){ if(!(hs_bp>0)) return 0; int b=(int)floor((log10(hs_bp)+2.0)*50.0); if(b<0)b=0; if(b>=NB)b=NB-1; return b; }
/* accumulate state (bid,ask) over [t0,t1) ns-since-midnight, clipped to RTH */
static void st_acc(Stats *s, long long t0, long long t1, double bid, double ask){
  if(t0<RTH0) t0=RTH0; if(t1>RTH1) t1=RTH1; if(t1<=t0) return;
  double dt=(double)(t1-t0)*1e-9;
  if(!(bid>0 && ask>0)) { s->invalid_t+=dt; return; }
  if(ask<bid){ s->invalid_t+=dt; return; }
  if(ask==bid){ s->locked_t+=dt; return; }
  double mid=0.5*(bid+ask), hs=(ask-bid)/(2*mid);
  if(hs>0.25){ s->invalid_t+=dt; return; }
  s->tw_t+=dt; s->tw_hs+=hs*dt; s->tw_dsp+=(ask-bid)*dt; s->tw_mid+=mid*dt;
  s->hist[hbin(hs*1e4)]+=dt;
  /* buckets */
  long long a=t0;
  while(a<t1){ int k=(int)((a-RTH0)/(300LL*1000000000LL)); if(k>=NBK)k=NBK-1; long long e=RTH0+(long long)(k+1)*300LL*1000000000LL; if(e>t1)e=t1; double d=(double)(e-a)*1e-9; s->bk_t[k]+=d; s->bk_hs[k]+=hs*d; a=e; }
}
static double hmed(const double *h, double tot){ double c=0; for(int i=0;i<NB;i++){ c+=h[i]; if(c>=0.5*tot) return pow(10.0,-2.0+(i+0.5)/50.0);} return NAN; }
static void st_header(FILE *f){
  fprintf(f,"symbol,rth_valid_s,invalid_s,locked_s,n_updates,tw_hs_bp,tw_med_hs_bp,tw_dollar_spread,tw_mid,tr_n,tr_dollar_vol,tr_shares,eff_hs_bp_dvw,eff_med_hs_bp,lit_dollar_vol,lit_eff_hs_bp_dvw");
  for(int k=0;k<NBK;k++) fprintf(f,",b%02d",k);
  fprintf(f,"\n");
}
static void st_write(FILE *f, const char *sym, Stats *s){
  if(s->tw_t<=0 && s->tr_n<=0) return;
  double tw=s->tw_t>0? s->tw_hs/s->tw_t*1e4 : NAN;
  double med=s->tw_t>0? hmed(s->hist,s->tw_t) : NAN;
  double tot=0; for(int i=0;i<NB;i++) tot+=s->tr_hist[i];
  fprintf(f,"%s,%.1f,%.1f,%.1f,%ld,%.4f,%.4f,%.6f,%.6f,%.0f,%.2f,%.0f,%.4f,%.4f,%.2f,%.4f",sym,s->tw_t,s->invalid_t,s->locked_t,s->nq,tw,med,
    s->tw_t>0?s->tw_dsp/s->tw_t:NAN, s->tw_t>0?s->tw_mid/s->tw_t:NAN, s->tr_n,s->tr_dv,s->tr_sh,
    s->tr_dv>0? s->tr_eff_dv/s->tr_dv*1e4:NAN, tot>0?hmed(s->tr_hist,tot):NAN, s->tr_lit_dv, s->tr_lit_dv>0? s->tr_lit_eff_dv/s->tr_lit_dv*1e4:NAN);
  for(int k=0;k<NBK;k++){ if(s->bk_t[k]>0) fprintf(f,",%.4f",s->bk_hs[k]/s->bk_t[k]*1e4); else fprintf(f,","); }
  fprintf(f,"\n");
}
