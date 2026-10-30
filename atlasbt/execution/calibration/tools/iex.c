/* IEX HIST TOPS (pcap/pcapng, uncompressed on stdin) -> per-symbol IEX-BBO spread stats.
   usage: zcat file.pcap.gz | iex UTC_OFFSET_HOURS > out.csv  (e.g. -4 for EDT) */
#include "common.h"
typedef struct { uint64_t key; long long last_t; double bid, ask; int avail; int etp; Stats s; char sym[9]; } Sym;
#define HS (1<<15)
static Sym *tab[HS];
static long long off_ns;
static Sym *get(const unsigned char *k8){ uint64_t k; memcpy(&k,k8,8); uint64_t h=(k*0x9E3779B97F4A7C15ULL)>>49; while(tab[h] && tab[h]->key!=k) h=(h+1)&(HS-1);
  if(!tab[h]){ Sym *s=calloc(1,sizeof(Sym)); s->key=k; memcpy(s->sym,k8,8); s->sym[8]=0; for(int i=7;i>=0 && s->sym[i]==' ';i--) s->sym[i]=0; s->last_t=-1; tab[h]=s; } return tab[h]; }
static long long day0=-1; /* ET midnight in epoch ns */
static long long et(long long ts){ long long t=ts+off_ns; if(day0<0){ day0=(t/86400000000000LL)*86400000000000LL; } return t-day0; }
static long nq=0, ntr=0;
static void msg(const unsigned char *m, int len){
  if(len<1) return;
  if(m[0]==0x51 && len>=42){ long long ts; memcpy(&ts,m+2,8); Sym *s=get(m+10); uint32_t bs,as; long long bp,ap; memcpy(&bs,m+18,4); memcpy(&bp,m+22,8); memcpy(&ap,m+30,8); memcpy(&as,m+38,4);
    long long t=et(ts); if(s->last_t>=0 && s->avail) st_acc(&s->s,s->last_t,t,s->bid,s->ask); else if(s->last_t>=0) st_acc(&s->s,s->last_t,t,0,0);
    s->last_t=t; s->bid=bp/1e4; s->ask=ap/1e4; s->avail=!(m[1]&0x80); s->s.nq++; nq++; }
  else if(m[0]==0x54 && len>=38){ long long ts; memcpy(&ts,m+2,8); Sym *s=get(m+10); uint32_t sz; long long px; memcpy(&sz,m+18,4); memcpy(&px,m+22,8);
    long long t=et(ts); double p=px/1e4; ntr++;
    if(t>=RTH0 && t<RTH1 && !(m[1]&0x40)){ s->s.tr_n+=1; s->s.tr_dv+=p*sz; s->s.tr_sh+=sz;
      if(s->bid>0 && s->ask>s->bid && s->avail){ double mid=0.5*(s->bid+s->ask), e=fabs(p-mid)/mid; if(e<0.05){ s->s.tr_eff_dv+=e*p*sz; s->s.tr_hist[hbin(e*1e4)]+=p*sz; s->s.tr_lit_dv+=p*sz; s->s.tr_lit_eff_dv+=e*p*sz; } } } }
  else if(m[0]==0x44 && len>=31){ Sym *s=get(m+10); s->etp=(m[1]&0x20)?1:0; }
}
static void pkt(const unsigned char *p, int caplen){
  int off=14; if(caplen<42) return; int eth=(p[12]<<8)|p[13]; if(eth==0x8100){ eth=(p[16]<<8)|p[17]; off=18; } if(eth!=0x0800) return;
  int ihl=(p[off]&15)*4; if(p[off+9]!=17) return; const unsigned char *pl=p+off+ihl+8; int plen=caplen-(off+ihl+8); if(plen<40) return;
  uint16_t cnt; memcpy(&cnt,pl+14,2); int pos=40; for(int i=0;i<cnt;i++){ if(pos+2>plen) break; uint16_t ml; memcpy(&ml,pl+pos,2); if(pos+2+ml>plen) break; msg(pl+pos+2,ml); pos+=2+ml; }
}
static unsigned char buf[1<<20];
int main(int argc,char**argv){
  off_ns=(long long)(atof(argc>1?argv[1]:"-4")*3600)*1000000000LL;
  FILE *f=stdin; uint32_t magic; if(fread(&magic,4,1,f)!=1) return 1;
  if(magic==0x0A0D0D0A){ /* pcapng */
    uint32_t blen; fread(&blen,4,1,f); fread(buf,1,blen-8,f);
    while(1){ uint32_t h[2]; if(fread(h,4,2,f)!=2) break; uint32_t bt=h[0], bl=h[1]; if(bl<12||bl-8>sizeof buf){fprintf(stderr,"bad block %u\n",bl);break;} if(fread(buf,1,bl-8,f)!=bl-8) break;
      if(bt==6){ uint32_t cl; memcpy(&cl,buf+12,4); pkt(buf+20,cl);} else if(bt==3){ uint32_t ol; memcpy(&ol,buf,4); pkt(buf+4,ol);} }
  } else if(magic==0xa1b2c3d4 || magic==0xa1b23c4d){
    fread(buf,1,20,f);
    while(1){ uint32_t r[4]; if(fread(r,4,4,f)!=4) break; uint32_t cl=r[2]; if(cl>sizeof buf) break; if(fread(buf,1,cl,f)!=cl) break; pkt(buf,cl); }
  } else { fprintf(stderr,"unknown magic %x\n",magic); return 1; }
  st_header(stdout);
  /* replace header: add etp column at end via separate file? simpler: print etp list to stderr-file */
  FILE *fe=fopen(argc>2?argv[2]:"/dev/null","w");
  for(int i=0;i<HS;i++) if(tab[i]){ Sym *s=tab[i]; if(s->last_t>=0) st_acc(&s->s,s->last_t,RTH1,s->avail?s->bid:0,s->avail?s->ask:0); st_write(stdout,s->sym,&s->s); if(fe) fprintf(fe,"%s,%d\n",s->sym,s->etp); }
  if(fe) fclose(fe);
  fprintf(stderr,"quotes=%ld trades=%ld\n",nq,ntr); return 0;
}
