"""Compile the actual LUT helper on CPU; CUDA scheduling is tested separately."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class LUTPrimitiveTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('c++'), 'requires a host C++ compiler')
    def test_all_interleaved_patterns_and_lut_recurrence(self):
        header = Path(__file__).resolve().parents[1] / 'src/fluxbin_style/csrc'
        code = r'''
#include "lut8.cuh"
#include <cmath>
#include <random>
int main() {
  for (unsigned word=0; word<65536; ++word) {
    for (unsigned basis=0; basis<2; ++basis) {
      unsigned expected=0;
      for (unsigned k=0; k<8; ++k)
        expected |= ((word>>(2*k+basis))&1u)<<k;
      if (fluxbin_lut::pattern(word>>basis)!=expected) return 1;
    }
  }
  std::mt19937 rng(19);
  std::uniform_real_distribution<float> dist(-32.f,32.f);
  for (int trial=0; trial<100; ++trial) {
    float x[8], lut[256];
    double norm=0.;
    for (int k=0; k<8; ++k) {
      x[k]=trial<8 ? (k==trial ? 1.f : 0.f) : dist(rng);
      norm+=std::abs(double(x[k]));
    }
    for (float &v:lut) v=NAN;
    for (unsigned lane=0; lane<32; ++lane)
      fluxbin_lut::build_lane(lut,x,lane);
    for (unsigned p=0; p<256; ++p) {
      double expected=0.;
      for (int k=0; k<8; ++k) expected+=((p>>k)&1u) ? double(x[k]) : -double(x[k]);
      if (!std::isfinite(lut[p]) || std::abs(double(lut[p])-expected)>2e-6*norm+1e-7)
        return 2;
    }
  }
}
'''
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'lut_test.cpp'
            binary = Path(folder)/'lut_test'
            source.write_text(code)
            subprocess.run(['c++','-std=c++17','-O2','-ffp-contract=off','-I',str(header),
                            str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            subprocess.run([str(binary)],check=True,capture_output=True,text=True)

    @unittest.skipUnless(shutil.which('c++'), 'requires a host C++ compiler')
    def test_unsigned_lut_recurrence(self):
        header = Path(__file__).resolve().parents[1] / 'src/fluxbin_style/csrc'
        code = r'''
#include "lut8_unsigned.cuh"
#include <cmath>
#include <random>
int main() {
  std::mt19937 rng(29);
  std::uniform_real_distribution<float> dist(-32.f,32.f);
  for (int trial=0; trial<100; ++trial) {
    float x[8], lut[256];
    double norm=0.;
    for (int k=0; k<8; ++k) { x[k]=dist(rng); norm+=std::abs(double(x[k])); }
    for (float &v:lut) v=NAN;
    for (unsigned lane=0; lane<32; ++lane)
      fluxbin_w3_lut::build_lane_unsigned(lut,x,lane);
    for (unsigned p=0; p<256; ++p) {
      double expected=0.;
      for (int k=0; k<8; ++k) if ((p>>k)&1u) expected+=double(x[k]);
      if (!std::isfinite(lut[p]) || std::abs(double(lut[p])-expected)>2e-6*norm+1e-7)
        return 1;
    }
  }
}
'''
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'unsigned_lut_test.cpp'
            binary = Path(folder)/'unsigned_lut_test'
            source.write_text(code)
            subprocess.run(['c++','-std=c++17','-O2','-ffp-contract=off','-I',str(header),
                            str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
            subprocess.run([str(binary)],check=True,capture_output=True,text=True)
