/* Daily TAQ NBBO + TRADE (pipe text, sorted by symbol,time) -> per-symbol spread stats.
   usage: taq NBBO_txt TRADE_txt > out.csv   (inputs may be pipes) */
#include "common.h"
#define MAXF 40
static int split(char *line, char **f){ int n=0; f[n++]=line; for(char *p=line;*p;p++){ if(*p=='|'){*p=0; if(n<MAXF) f[n++]=p+1;} else if(*p=='\n'||*p=='\r'){*p=0;} } return n; }
static long long ptime(const char *s){ /* HHMMSSnnnnnnnnn */
  if(strlen(s)<15) return -1; long long h=(s[0]-'0')*10+(s[1]-'0'), m=(s[2]-'0')*10+(s[3]-'0'), se=(s[4]-'0')*10+(s[5]-'0');
  long long ns=0; for(int i=6;i<15;i++) ns=ns*10+(s[i]-'0'); return ((h*3600+m*60+se)*1000000000LL)+ns; }
static long long *T; static double *BB,*BA; static long cap=0,n=0;
static void push(long long t,double b,double a){ if(n>=cap){ cap=cap?cap*2:1<<20; T=realloc(T,cap*sizeof(*T)); BB=realloc(BB,cap*sizeof(*BB)); BA=realloc(BA,cap*sizeof(*BA)); } T[n]=t;BB[n]=b;BA[n]=a;n++; }
static int badcond(const char *c){ for(;*c;c++){ if(strchr("TUZLCNRWPBGHMQO4567893V",*c)) return 1; } return 0; }
static FILE *ft; static char tline[4096]; static char *tf[MAXF]; static int thave=0, tn=0; static char tsym[64];
static long tr_skipped=0;
static int tnext(void){ while(fgets(tline,sizeof tline,ft)){ tn=split(tline,tf); if(tn<9) continue; if(strcmp(tf[0],"Time")==0) continue; strncpy(tsym,tf[2],63); thave=1; return 1;} thave=0; return 0; }
static void finish(const char *sym){
  Stats s; st_reset(&s); s.nq=n;
  for(long i=0;i<n;i++){ long long t1=(i+1<n)?T[i+1]:RTH1; if(T[i]<RTH1 && t1>RTH0) st_acc(&s,T[i],t1,BB[i],BA[i]); }
  /* trades */
  while(thave && strcmp(tsym,sym)<0){ tr_skipped++; tnext(); }
  while(thave && strcmp(tsym,sym)==0){
    long long t=ptime(tf[0]);
    if(t>=RTH0 && t<RTH1 && strcmp(tf[7],"00")==0 && !badcond(tf[3])){
      double q=atof(tf[4]), p=atof(tf[5]);
      if(q>0 && p>0){
        /* last NBBO strictly before t */
        long lo=0,hi=n; while(lo<hi){ long md=(lo+hi)/2; if(T[md]<t) lo=md+1; else hi=md; }
        long j=lo-1;
        s.tr_n+=1; s.tr_dv+=p*q; s.tr_sh+=q;
        if(j>=0 && BB[j]>0 && BA[j]>BB[j]){
          double m=0.5*(BB[j]+BA[j]); double e=fabs(p-m)/m;
          if(e<0.05){ s.tr_eff_dv+=e*p*q; s.tr_eff_n+=1; s.tr_hist[hbin(e*1e4)]+=p*q;
            if(tf[1][0]!='D'){ s.tr_lit_dv+=p*q; s.tr_lit_eff_dv+=e*p*q; } }
        }
      }
    }
    tnext();
  }
  /* eff spread dvw: only over trades with valid nbbo -> renormalise */
  double dv_valid=0; for(int i=0;i<NB;i++) dv_valid+=s.tr_hist[i];
  if(dv_valid>0){ double r=s.tr_eff_dv/dv_valid; double dv=s.tr_dv; s.tr_eff_dv=r*dv; }
  st_write(stdout,sym,&s);
}
int main(int argc,char**argv){
  if(argc<3){fprintf(stderr,"usage\n");return 1;}
  FILE *fq=fopen(argv[1],"r"); ft=fopen(argv[2],"r"); if(!fq||!ft){perror("open");return 1;}
  if(!(argc>3)) st_header(stdout); tnext();
  static char line[4096]; char *f[MAXF]; char cur[64]=""; long rows=0;
  const char *after = argc>3 ? argv[3] : "";
  while(fgets(line,sizeof line,fq)){
    if(after[0]){ char *bar1=strchr(line,'|'); char *bar2=bar1?strchr(bar1+1,'|'):0; char *bar3=bar2?strchr(bar2+1,'|'):0;
      if(bar3){ size_t L=bar3-bar2-1; char tmp[64]; if(L>63)L=63; memcpy(tmp,bar2+1,L); tmp[L]=0; if(strcmp(tmp,after)<=0) continue; } }
    int nf=split(line,f); if(nf<23) continue; if(strcmp(f[0],"Time")==0) continue;
    if(strcmp(f[2],cur)!=0){ if(cur[0]) finish(cur); strncpy(cur,f[2],63); n=0; }
    if(f[12][0]=='B' || f[12][0]=='C' ) { } /* cancel/correction flags: keep NBBO as published */
    long long t=ptime(f[0]); if(t<0) continue;
    push(t, atof(f[16]), atof(f[21]));
    if(++rows % 100000000 == 0) fprintf(stderr,"%ld rows, sym %s\n",rows,cur);
  }
  if(cur[0]) finish(cur);
  fprintf(stderr,"done rows=%ld trades_skipped_nosym=%ld\n",rows,tr_skipped);
  return 0;
}
