#include "../../../cuda/kernel.cu"
#include <cassert>
#include <vector>
#include <algorithm>

using u128 = unsigned __int128;
static u64 state = 0x61e5d773b942a019ull;
static u64 random64() { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state; }
static u64 inverse64(u64 x) { u64 z=1; for (int i=0;i<6;i++) z*=2-x*z; return z; }
static u128 from_words(const u32* x) { return (u128)x[0] | (u128)x[1]<<32 | (u128)x[2]<<64 | (u128)x[3]<<96; }
static void to_words(u128 x,u32* w) { for(int j=0;j<4;j++)w[j]=(u32)(x>>(32*j)); }
static unsigned long long tested_words=0,tested_bits=0,launches=0;
static void check(void* ctx, u32 mode, const std::vector<u32>& input,
                  const std::vector<u32>& expected, KParams p, u64 magic,
                  const u32* tp,const std::vector<u32>& bv,const std::vector<u32>& mod9) {
    const u32 n=expected.size()/8;
    assert(cv_gpu_launch_screen_ex(ctx,0,input.data(),n,n,mode,0,magic,tp,&p)==0);
    assert(cv_gpu_launch_screen_ex(ctx,0,input.data(),n,n,mode,0,magic,tp,&p)==-1);
    assert(cv_gpu_prepare_screen(ctx,p.words32)==-1);
    float ms=0,copy_ms=0; unsigned long long cases=0,iters=0;u32 hash=0,flags=0;
    assert(cv_gpu_wait(ctx,0,&ms,&cases,&iters,&hash,&flags,nullptr,0)==-1);
    const u32 *affine=nullptr,*masks=nullptr; u32 got_n=0,got_words=0;
    assert(cv_gpu_wait_screen(ctx,0,&ms,&copy_ms,&affine,&masks,&got_n,&got_words)==0);
    assert(got_n==n&&got_words==p.words32&&ms>=0&&copy_ms>=0);
    assert(std::equal(expected.begin(),expected.end(),affine));
    const u128 p3=from_words(tp);
    for(u32 i=0;i<n;i++) {
        u32 basewords[4];for(int j=0;j<4;j++)basewords[j]=expected[8*i+j];
        const u32 oee=basewords[3]>>30,family=(basewords[3]>>29)&1u;
        basewords[3]&=0x1fffffffu;
        const u128 base=from_words(basewords),m=from_words(&expected[8*i+4]);
        const u32 r9=(u32)(base%9);
        const u64 index=magic*(u64)(m+1)&p.b_mask;
        for(u32 w=0;w<p.words32;w++) {
            u32 want=0;
            for(u32 b=0;b<32;b++) {
                const u32 a=w*32+b;
                const u64 j=(index+a)&p.b_mask;
                const bool bv_ok=(bv[family*p.fam_stride+(j>>5)]>>(j&31))&1;
                const bool m9_ok=(mod9[r9*p.words32+w]>>b)&1;
                const u128 iterate=m+p3*a;
                const bool oee_ok=!p.tail_oee || !((oee==2&&!(iterate&1)) || (oee==1&&!(iterate&3)));
                if(bv_ok&&m9_ok&&oee_ok)want|=1u<<b;
            }
            assert(masks[(size_t)i*p.words32+w]==want);
            tested_words++; tested_bits+=32;
        }
    }
    assert(cv_gpu_wait_screen(ctx,0,&ms,&copy_ms,&affine,&masks,&got_n,&got_words)==-1);
    launches++;
}
int main() {
 for(u32 a: {8u,10u}) for(u32 m: {62u,64u}) for(u32 families: {1u,2u}) {
    constexpr u32 b=12, cap=300;
    const u32 words=(1u<<a)/32,base_words=(1u<<b)/32,stride=base_words+words;
    std::vector<u32> bv((size_t)stride*families),mod9(9*words);
    for(u32 fam=0;fam<families;fam++) {
        for(u32 j=0;j<base_words;j++)bv[fam*stride+j]=(u32)random64();
        for(u32 j=0;j<words;j++)bv[fam*stride+base_words+j]=bv[fam*stride+j];
    }
    for(auto& v:mod9)v=(u32)random64();
    void* ctx=cv_gpu_init(bv.data(),bv.size(),stride,mod9.data(),mod9.size(),2,cap);
    assert(ctx);assert(cv_gpu_enable_experimental(ctx,cap)==0);
    if(families==2)assert(cv_gpu_prepare_screen(ctx,words)==0);
    u128 p3=1;for(int i=0;i<40;i++)p3*=3;
    u32 tp[4];to_words(p3,tp);const u64 magic=inverse64((u64)p3);
    for(u32 n: {1u,7u,8u,9u,17u,33u,257u}) for(u32 enabled: {0u,1u}) {
        KParams p={};p.b_mask=(1u<<b)-1;p.fam_stride=families==2?stride:0;
        p.m=m;p.m_limb=m/32;p.m_shift=m%32;p.words32=words;p.tail_oee=enabled;
        p.pad=(n%3==0?1u:n%3==1?0u:2u)<<8;
        std::vector<u32> affine(8*n);
        for(u32 i=0;i<n;i++) {
            const u64 base=random64()&(m==64?~0ull:((1ull<<m)-1));
            to_words(base,&affine[8*i]);
            affine[8*i+3]|=((i%3)<<30)|((i%families)<<29);
            u128 value=(u128)(random64()&((1ull<<40)-1))<<64;
            const u64 edge[]={0,31,(1u<<b)-1,(1u<<b)-8};
            const u64 lo=i%5==4?random64():(u64)p3*edge[i%4]-1;
            value|=lo;to_words(value,&affine[8*i+4]);
        }
        check(ctx,2,affine,affine,p,magic,tp,bv,mod9);
    }
    // Two even affine suffix steps, independently reconstructed on CPU.
    const u32 terminal[]={0,0,0,1,40,0,2,0};
    assert(cv_gpu_upload_terminal(ctx,terminal,1)==0);
    const u32 program[]={0,1,1,1,1,0,1,0},leaf[]={0,1};
    const u32 chainwords[]={0,0,0,1,40,0,1,0,0,0,0,1,40,0,2,0};
    assert(cv_gpu_upload_chain(ctx,program,1,leaf,1,chainwords,2)==0);
    for(u32 mode: {3u,4u}) {
        const u32 n=17,k=m-2;
        std::vector<u32> parents(16*n),expected(8*n);
        for(u32 i=0;i<n;i++) {
            const u64 n0=random64()&((1ull<<k)-1);
            const u128 iterate=((u128)(random64()&((1ull<<20)-1))<<64)|random64();
            u32* d=&parents[16*i];d[0]=(u32)n0;d[1]=n0>>32;to_words(iterate,d+2);to_words(p3,d+6);
            const u32 delta=(u32)((u64)iterate*magic)&3;
            d[10]=mode==3?delta:(u32)magic;d[11]=0;d[12]=1;d[13]=i;d[14]=mode==3?2:0;d[15]=k;
            const u32 q=(0u-delta)&3;
            const u128 x=iterate+p3*q;assert((x&3)==0);
            to_words((u128)n0|((u128)q<<k),&expected[8*i]);to_words(x>>2,&expected[8*i+4]);
        }
        KParams p={};p.b_mask=(1u<<b)-1;p.fam_stride=families==2?stride:0;p.m=m;p.words32=words;p.tail_oee=1;p.pad=5;
        check(ctx,mode,parents,expected,p,magic,tp,bv,mod9);
    }
    cv_gpu_free(ctx);
 }
 printf("{\"mask_oracle\":\"passed\",\"launches\":%llu,\"mask_words\":%llu,\"completion_bits\":%llu,\"a\":[8,10],\"m\":[62,64],\"families\":[1,2],\"oee_states\":[0,1,2],\"modes\":[2,3,4],\"blocks\":[128,256,512]}\n",launches,tested_words,tested_bits);
}
