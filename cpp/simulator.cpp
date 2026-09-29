// Functional interpreter for TILEAC01. Timing is modeled separately.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t max_elements=16000000, max_instructions=1000000;
uint32_t read32(const std::vector<uint8_t>& b,size_t at) {
  if(at+4>b.size()) throw std::runtime_error("truncated uint32");
  return uint32_t(b[at])|(uint32_t(b[at+1])<<8)|(uint32_t(b[at+2])<<16)|(uint32_t(b[at+3])<<24);
}
float readfloat(const std::vector<uint8_t>& b,size_t at) {
  auto bits=read32(b,at);float v;std::memcpy(&v,&bits,4);return v;
}
// IEEE binary16 round-to-nearest, ties-to-even, carried in float32.
float round_half(float v) {
  uint32_t bits;std::memcpy(&bits,&v,4);
  uint32_t sign=(bits>>16)&0x8000u,mant=bits&0x7fffffu;
  int exp=int((bits>>23)&255u)-112;
  uint32_t half;
  if(((bits>>23)&255u)==255u) return v;
  if(exp<=0) {
    if(exp < -10) half=sign;
    else {
      mant|=0x800000u;int shift=14-exp;
      uint32_t value=mant>>shift,rem=mant&((1u<<shift)-1u),tie=1u<<(shift-1);
      if(rem>tie || (rem==tie && (value&1u))) ++value;
      half=sign|value;
    }
  } else {
    uint32_t value=mant>>13,rem=mant&8191u;
    if(rem>4096u || (rem==4096u && (value&1u))) ++value;
    if(value==1024u) {++exp;value=0;}
    half=exp>=31?sign|0x7c00u:sign|(uint32_t(exp)<<10)|value;
  }
  uint32_t he=(half>>10)&31u,hm=half&1023u;
  float result=he==0?std::ldexp(float(hm),-24):he==31?std::numeric_limits<float>::infinity():std::ldexp(1.f+float(hm)/1024.f,int(he)-15);
  return (half&0x8000u)?-result:result;
}
std::vector<uint8_t> readfile(const std::string& path) {
  std::ifstream f(path,std::ios::binary|std::ios::ate);
  if(!f) throw std::runtime_error("cannot open "+path);
  auto n=f.tellg();if(n<0 || n>128000036) throw std::runtime_error("invalid file size");
  std::vector<uint8_t> bytes(static_cast<size_t>(n));f.seekg(0);f.read(reinterpret_cast<char*>(bytes.data()),n);
  if(!f) throw std::runtime_error("short read");
  return bytes;
}
void range(uint64_t offset,uint64_t size,uint64_t limit) {
  if(offset>limit || size>limit-offset) throw std::runtime_error("address outside buffer");
}
void matrix(uint64_t offset,uint64_t rows,uint64_t cols,uint64_t stride,uint64_t limit) {
  if(!rows || !cols || stride<cols) throw std::runtime_error("invalid matrix extent/stride");
  range(offset,(rows-1)*stride+cols,limit);
}
void writefloat(std::ofstream& f,float v) {
  uint32_t bits;std::memcpy(&bits,&v,4);
  uint8_t b[4]={uint8_t(bits),uint8_t(bits>>8),uint8_t(bits>>16),uint8_t(bits>>24)};
  f.write(reinterpret_cast<char*>(b),4);
}
}

int main(int argc,char** argv) {
  try {
    if(argc!=4) throw std::runtime_error("usage: tile-sim PROGRAM INPUT_F32 OUTPUT_F32");
    auto blob=readfile(argv[1]);
    if(blob.size()<36 || std::memcmp(blob.data(),"TILEAC01",8)!=0 || read32(blob,8)!=1)
      throw std::runtime_error("unsupported binary header");
    auto count=read32(blob,12), mem_count=read32(blob,16), scratch_count=read32(blob,20);
    auto input_count=read32(blob,24), out_offset=read32(blob,28), out_count=read32(blob,32);
    if(!count || count>max_instructions || !mem_count || mem_count>max_elements ||
       !scratch_count || scratch_count>max_elements || !input_count || input_count>mem_count || !out_count)
      throw std::runtime_error("invalid resource limits");
    range(out_offset,out_count,mem_count);
    uint64_t payload=36+uint64_t(count)*64;
    if(blob.size()!=payload+uint64_t(mem_count)*4) throw std::runtime_error("binary length mismatch");
    if(read32(blob,36+uint64_t(count-1)*64)!=8) throw std::runtime_error("missing final HALT");
    std::vector<float> memory(mem_count),scratch(scratch_count);
    std::vector<bool> initialized(scratch_count,false),events(count+1,false);
    events[0]=true;
    for(uint32_t ix=0;ix<mem_count;++ix) {
      memory[ix]=readfloat(blob,payload+uint64_t(ix)*4);
      if(!std::isfinite(memory[ix])) throw std::runtime_error("nonfinite program constant");
    }
    auto input=readfile(argv[2]);
    if(input.size()!=uint64_t(input_count)*4) throw std::runtime_error("input size mismatch");
    for(uint32_t ix=0;ix<input_count;++ix) {
      memory[ix]=readfloat(input,uint64_t(ix)*4);
      if(!std::isfinite(memory[ix])) throw std::runtime_error("nonfinite input");
    }
    auto require=[&](uint64_t offset,uint64_t size) {
      range(offset,size,scratch_count);
      for(uint64_t i=offset;i<offset+size;++i) if(!initialized[i]) throw std::runtime_error("uninitialized scratch read");
    };
    auto mark=[&](uint64_t offset,uint64_t size) {
      range(offset,size,scratch_count);for(uint64_t i=offset;i<offset+size;++i) initialized[i]=true;
    };
    uint64_t multiply_adds=0,dma_bytes=0;
    for(uint32_t pc=0;pc<count;++pc) {
      uint32_t f[16];for(size_t j=0;j<16;++j) f[j]=read32(blob,36+uint64_t(pc)*64+j*4);
      auto op=f[0],a=f[1],b=f[2],c=f[3],m=f[4],n=f[5],k=f[6],flags=f[7],event=f[8],dep=f[9],stride=f[10];
      if(event!=pc+1 || dep>pc || !events[dep]) throw std::runtime_error("invalid event dependency");
      if(f[12] || f[13] || f[14] || f[15]) throw std::runtime_error("nonzero reserved fields");
      if(op==1) { // Host memory -> compact scratch tile.
        if(flags&~1u) throw std::runtime_error("invalid LOAD flags");
        matrix(a,m,n,stride,mem_count);range(b,uint64_t(m)*n,scratch_count);
        for(uint32_t r=0;r<m;++r) for(uint32_t col=0;col<n;++col) {
          float value=memory[a+uint64_t(r)*stride+col];
          if(flags&1u) value=round_half(value);
          if(!std::isfinite(value)) throw std::runtime_error("operand conversion overflow");
          scratch[b+uint64_t(r)*n+col]=value;
        }
        mark(b,uint64_t(m)*n);dma_bytes+=uint64_t(m)*n*4;
      } else if(op==2) {
        if(flags) throw std::runtime_error("invalid STORE flags");
        require(a,uint64_t(m)*n);matrix(b,m,n,stride,mem_count);
        for(uint32_t r=0;r<m;++r) for(uint32_t col=0;col<n;++col) memory[b+uint64_t(r)*stride+col]=scratch[a+uint64_t(r)*n+col];
        dma_bytes+=uint64_t(m)*n*4;
      } else if(op==3) {
        if(flags || !m) throw std::runtime_error("invalid ZERO");
        range(a,m,scratch_count);std::fill(scratch.begin()+a,scratch.begin()+a+m,0.f);mark(a,m);
      } else if(op==4) {
        if(flags&~1u || !m || !n || !k) throw std::runtime_error("invalid MATMUL");
        if(f[11]>pc || !events[f[11]]) throw std::runtime_error("invalid accumulator dependency");
        require(a,uint64_t(m)*k);require(b,uint64_t(k)*n);range(c,uint64_t(m)*n,scratch_count);
        if((uint64_t(c)<uint64_t(a)+uint64_t(m)*k && uint64_t(a)<uint64_t(c)+uint64_t(m)*n) ||
           (uint64_t(c)<uint64_t(b)+uint64_t(k)*n && uint64_t(b)<uint64_t(c)+uint64_t(m)*n))
          throw std::runtime_error("MATMUL output aliases input");
        if(flags&1u) require(c,uint64_t(m)*n);
        for(uint32_t r=0;r<m;++r) for(uint32_t col=0;col<n;++col) {
          float sum=(flags&1u)?scratch[c+uint64_t(r)*n+col]:0.f;
          for(uint32_t kk=0;kk<k;++kk) sum+=scratch[a+uint64_t(r)*k+kk]*scratch[b+uint64_t(kk)*n+col];
          if(!std::isfinite(sum)) throw std::runtime_error("nonfinite matrix accumulation");
          scratch[c+uint64_t(r)*n+col]=sum;
        }
        mark(c,uint64_t(m)*n);multiply_adds+=uint64_t(m)*n*k;
      } else if(op==5) {
        if(flags&~3u || !m || !n) throw std::runtime_error("invalid EPILOGUE");
        require(a,uint64_t(m)*n);require(b,n);if(flags&2u) require(c,uint64_t(m)*n);
        for(uint32_t r=0;r<m;++r) for(uint32_t col=0;col<n;++col) {
          auto ix=uint64_t(r)*n+col;float v=scratch[a+ix]+scratch[b+col];if(flags&2u) v+=scratch[c+ix];
          if(!std::isfinite(v)) throw std::runtime_error("nonfinite epilogue");
          scratch[a+ix]=(flags&1u)?std::max(0.f,v):v;
        }
      } else if(op==6) {
        if(flags) throw std::runtime_error("invalid WAIT");
      } else if(op==7) { // Optional native 3x3 depthwise unit; HWC data.
        if(flags&~3u || !m || !n || !k) throw std::runtime_error("invalid DEPTHWISE");
        range(a,uint64_t(m)*n*k,mem_count);range(b,uint64_t(k)*9,mem_count);
        range(c,uint64_t(m)*n*k,mem_count);range(f[11],k,mem_count);
        if(uint64_t(c)<uint64_t(a)+uint64_t(m)*n*k && uint64_t(a)<uint64_t(c)+uint64_t(m)*n*k)
          throw std::runtime_error("depthwise output aliases input");
        for(uint32_t y=0;y<m;++y) for(uint32_t x=0;x<n;++x) for(uint32_t ch=0;ch<k;++ch) {
          float sum=memory[f[11]+ch];
          for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
            int iy=int(y)+dy,ix=int(x)+dx;
            if(iy>=0 && ix>=0 && iy<int(m) && ix<int(n)) {
              float av=memory[a+(uint64_t(iy)*n+ix)*k+ch],bv=memory[b+uint64_t(ch)*9+(dy+1)*3+dx+1];
              if(flags&2u) {av=round_half(av);bv=round_half(bv);}
              if(!std::isfinite(av) || !std::isfinite(bv)) throw std::runtime_error("depthwise operand conversion overflow");
              sum+=av*bv;
              ++multiply_adds;
            }
          }
          if(!std::isfinite(sum)) throw std::runtime_error("nonfinite depthwise accumulation");
          memory[c+(uint64_t(y)*n+x)*k+ch]=(flags&1u)?std::max(0.f,sum):sum;
        }
      } else if(op==8) {
        if(pc!=count-1 || flags) throw std::runtime_error("invalid HALT placement");
      } else throw std::runtime_error("unknown opcode");
      events[event]=true;
    }
    std::ofstream output(argv[3],std::ios::binary|std::ios::trunc);
    if(!output) throw std::runtime_error("cannot create output");
    for(uint32_t i=0;i<out_count;++i) {
      auto value=memory[out_offset+i];if(!std::isfinite(value)) throw std::runtime_error("nonfinite result");
      writefloat(output,value);
    }
    if(!output) throw std::runtime_error("output write failed");
    std::cout<<"{\"instructions\":"<<count<<",\"multiply_adds\":"<<multiply_adds<<",\"dma_bytes\":"<<dma_bytes<<"}\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr<<"tile-sim: "<<e.what()<<"\n";return 2;
  }
}
