// Copyright 2017 ProjectQ-Framework (www.projectq.ch)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


template <class V>
inline void kernel_core(V &psi, std::size_t I, std::size_t d0, __m256d m0, __m256d m1, __m256d m2, __m256d m3)
{
  // tell the compiler that d0 > 0
  if( d0 == 0 )
    return;

  /*
     std::complex<double> v[2];
     v0 = psi[I];
     v1 = psi[I + d0];

     psi[I] = v0 * m00 + v1*m01;
     psi[I + d0] = v0 * m10 + v1 * m11;
  // with complex multiplication
  psi[I].r =    v0.r * m00.r - v0.i * m00.i + v1.r * m01.r - v1.i * m01.i
  psi[I].i =    v0.r * m00.i + v0.i * m00.r + v1.r * m01.i + v1.i * m01.r
  psi[I+d0].r = v0.r * m10.r - v0.i * m10.i + v1.r * m11.r - v1.i * m11.i
  psi[I+d0].i = v0.r * m10.i + v0.i * m10.r + v1.r * m11.i + v1.i * m11.r
  */

  double* psiI = (double*)&(psi[I]);
  double* psiId0 = (double*)&(psi[I+d0]);
  __m256d v0r = _mm256_broadcast_sd(psiI);
  __m256d v0i = _mm256_broadcast_sd(psiI+1);
  __m256d v1r = _mm256_broadcast_sd(psiId0);
  __m256d v1i = _mm256_broadcast_sd(psiId0+1);

  __m256d tmp1 = _mm256_mul_pd(v0r, m0);
  __m256d tmp2 = _mm256_mul_pd(v0i, m1);
  __m256d tmp3 = _mm256_fmadd_pd(v1r, m2, tmp1);
  __m256d tmp4 = _mm256_fmadd_pd(v1i, m3, tmp2);
  __m256d tmp5 = _mm256_add_pd(tmp3, tmp4);

  __m128d new_v0 = _mm256_extractf128_pd(tmp5, 0);
  __m128d new_v1 = _mm256_extractf128_pd(tmp5, 1);

  _mm_store_pd(psiI, new_v0);
  _mm_store_pd(psiId0, new_v1);
}

template <class V>
inline void kernel_core1(V &psi, std::size_t I, __m256d m0, __m256d m1, __m256d m2, __m256d m3)
{
  // d0 == 1
  double* psiI = (double*)&(psi[I]);
  double* psiId0 = (double*)&(psi[I+1]);
  __m256d v0r = _mm256_broadcast_sd(psiI);
  __m256d v0i = _mm256_broadcast_sd(psiI+1);
  __m256d v1r = _mm256_broadcast_sd(psiId0);
  __m256d v1i = _mm256_broadcast_sd(psiId0+1);

  __m256d tmp1 = _mm256_mul_pd(v0r, m0);
  __m256d tmp2 = _mm256_mul_pd(v0i, m1);
  __m256d tmp3 = _mm256_fmadd_pd(v1r, m2, tmp1);
  __m256d tmp4 = _mm256_fmadd_pd(v1i, m3, tmp2);
  __m256d tmp5 = _mm256_add_pd(tmp3, tmp4);

  _mm256_store_pd(psiI, tmp5);
}

template <class V>
inline void kernel_core_unroll2(V &psi, std::size_t I, std::size_t d0, __m256d m0, __m256d m1, __m256d m2, __m256d m3)
{
  // d0 > 1, so we do two iterations at once!
  // tell the compiler that d0 > 0
  if( d0 <= 1 )
    return;

  // first iteration
  double* psiI = (double*)&(psi[I]);
  double* psiId0 = (double*)&(psi[I+d0]);
  __m256d v0r = _mm256_broadcast_sd(psiI);
  __m256d v0i = _mm256_broadcast_sd(psiI+1);
  __m256d v1r = _mm256_broadcast_sd(psiId0);
  __m256d v1i = _mm256_broadcast_sd(psiId0+1);

  __m256d tmp1 = _mm256_mul_pd(v0r, m0);
  __m256d tmp2 = _mm256_mul_pd(v0i, m1);
  __m256d tmp3 = _mm256_fmadd_pd(v1r, m2, tmp1);
  __m256d tmp4 = _mm256_fmadd_pd(v1i, m3, tmp2);
  __m256d result1 = _mm256_add_pd(tmp3, tmp4);

  // second iteration
  v0r = _mm256_broadcast_sd(psiI+2);
  v0i = _mm256_broadcast_sd(psiI+3);
  v1r = _mm256_broadcast_sd(psiId0+2);
  v1i = _mm256_broadcast_sd(psiId0+3);

  tmp1 = _mm256_mul_pd(v0r, m0);
  tmp2 = _mm256_mul_pd(v0i, m1);
  tmp3 = _mm256_fmadd_pd(v1r, m2, tmp1);
  tmp4 = _mm256_fmadd_pd(v1i, m3, tmp2);
  __m256d result2 = _mm256_add_pd(tmp3, tmp4);

  __m128d new_v00 = _mm256_extractf128_pd(result1, 0);
  __m128d new_v01 = _mm256_extractf128_pd(result1, 1);
  __m128d new_v10 = _mm256_extractf128_pd(result2, 0);
  __m128d new_v11 = _mm256_extractf128_pd(result2, 1);

  __m256d new_v0 = _mm256_set_m128d(new_v10, new_v00);
  __m256d new_v1 = _mm256_set_m128d(new_v11, new_v01);

  _mm256_store_pd(psiI, new_v0);
  _mm256_store_pd(psiId0, new_v1);
}


// bit indices id[.] are given from high to low (e.g. control first for CNOT)
template <class V, class M>
void kernel(V &psi, unsigned id0, M const& m, std::size_t ctrlmask)
{
  std::size_t n = psi.size();
  std::size_t d0 = 1UL << id0;
  const std::size_t nOuter = n / (2*d0);

  // see comments in kernel function above
  //  m0=         m1=        m2=         m3=
  // m00.r     - m00.i      m01.r     - m01.i
  // m00.i     + m00.r      m01.i     + m01.r
  // m10.r     - m10.i      m11.r     - m11.i
  // m10.i     + m10.r      m11.i     + m11.r
  __m256d tmp_m0 = _mm256_set_pd(m[1][0].imag(), m[1][0].real(), m[0][0].imag(), m[0][0].real());
  __m256d tmp_m1 = _mm256_set_pd(m[1][0].real(), -m[1][0].imag(), m[0][0].real(), -m[0][0].imag());
  __m256d tmp_m2 = _mm256_set_pd(m[1][1].imag(), m[1][1].real(), m[0][1].imag(), m[0][1].real());
  __m256d tmp_m3 = _mm256_set_pd(m[1][1].real(), -m[1][1].imag(), m[0][1].real(), -m[0][1].imag());


  if (ctrlmask == 0)
  {
    // unroll
    if (d0 == 1)
    {
#pragma omp for schedule(static)
      for (std::size_t iOuter = 0; iOuter < nOuter; iOuter++)
      {
        std::size_t i0 = iOuter * 2;
        kernel_core1(psi, i0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
      }
    }
    else if (d0 == 2)
    {
#pragma omp for schedule(static)
      for (std::size_t iOuter = 0; iOuter < nOuter; iOuter++)
      {
        std::size_t i0 = iOuter * 4;
        kernel_core_unroll2(psi, i0+0, 2, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
        //kernel_core(psi, i0+1, 2, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
      }
    }
    else if (d0 == 4)
    {
#pragma omp for schedule(static)
      for (std::size_t iOuter = 0; iOuter < nOuter; iOuter++)
      {
        std::size_t i0 = iOuter * 8;
        kernel_core_unroll2(psi, i0+0, 4, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
        //kernel_core(psi, i0+1, 4, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
        kernel_core_unroll2(psi, i0+2, 4, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
        //kernel_core(psi, i0+3, 4, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
      }
    }
    else // at least 8
    {
      const std::size_t nInner = d0/8;
      if( nOuter > nInner )
      {
#pragma omp for schedule(static)
        for (std::size_t iOuter = 0; iOuter < nOuter; iOuter++)
          for (std::size_t iInner = 0; iInner < nInner; iInner++)
          {
            std::size_t i0 = iOuter * 2*d0;
            std::size_t i1 = iInner * 8;
            kernel_core_unroll2(psi, i0+i1+0, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            //kernel_core(psi, i0+i1+1, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            kernel_core_unroll2(psi, i0+i1+2, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            //kernel_core(psi, i0+i1+3, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            kernel_core_unroll2(psi, i0+i1+4, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            //kernel_core(psi, i0+i1+5, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            kernel_core_unroll2(psi, i0+i1+6, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            //kernel_core(psi, i0+i1+7, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
          }
      }
      else // nInner > nOuter
      {
        for (std::size_t iOuter = 0; iOuter < nOuter; iOuter++)
        {
#pragma omp for schedule(static) nowait
          for (std::size_t iInner = 0; iInner < nInner; iInner++)
          {
            std::size_t i0 = iOuter * 2*d0;
            std::size_t i1 = iInner * 8;
            kernel_core_unroll2(psi, i0+i1+0, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            //kernel_core(psi, i0+i1+1, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            kernel_core_unroll2(psi, i0+i1+2, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            //kernel_core(psi, i0+i1+3, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            kernel_core_unroll2(psi, i0+i1+4, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            //kernel_core(psi, i0+i1+5, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            kernel_core_unroll2(psi, i0+i1+6, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            //kernel_core(psi, i0+i1+7, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
          }
        }
      }
    }
  }
  else
  {
    // unroll
    if (d0 == 1)
    {
#pragma omp for schedule(static,96)
      for (std::size_t iOuter = 0; iOuter < nOuter; iOuter++)
      {
        std::size_t i0 = iOuter * 2;
        if ( (i0 & ctrlmask) == ctrlmask) kernel_core1(psi, i0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
      }
    }
    else if (d0 == 2)
    {
#pragma omp for schedule(static,48)
      for (std::size_t iOuter = 0; iOuter < nOuter; iOuter++)
      {
        std::size_t i0 = iOuter * 4;
        if ( ((i0+0) & ctrlmask) == ctrlmask) kernel_core(psi, i0+0, 2, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
        if ( ((i0+1) & ctrlmask) == ctrlmask) kernel_core(psi, i0+1, 2, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
      }
    }
    else if (d0 == 4)
    {
#pragma omp for schedule(static,24)
      for (std::size_t iOuter = 0; iOuter < nOuter; iOuter++)
      {
        std::size_t i0 = iOuter * 8;
        if ( ((i0+0) & ctrlmask) == ctrlmask) kernel_core(psi, i0+0, 4, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
        if ( ((i0+1) & ctrlmask) == ctrlmask) kernel_core(psi, i0+1, 4, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
        if ( ((i0+2) & ctrlmask) == ctrlmask) kernel_core(psi, i0+2, 4, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
        if ( ((i0+3) & ctrlmask) == ctrlmask) kernel_core(psi, i0+3, 4, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
      }
    }
    else // at least 8
    {
      const std::size_t nInner = d0/8;
      //std::cout << " ctrlmask: " << ctrlmask << " (" << __builtin_popcountll(ctrlmask) << "), d0: " << d0 << std::endl;
      // special case for single-bit control mask > d0...
      if( __builtin_popcountll(ctrlmask) == 1 && ctrlmask > d0 )
      {
        //std::cout << " ctrlmask: " << ctrlmask << " (" << __builtin_popcountll(ctrlmask) << "), d0: " << d0 << ", nOuter: " << nOuter << std::endl;
        // parallelize over inner loop
        if( nOuter < nInner )
        {
          for (std::size_t iOuter = 0; iOuter < nOuter; iOuter++)
          {
            std::size_t i0 = iOuter * 2*d0;
            if ( (i0 & ctrlmask) != ctrlmask)
              continue;
#pragma omp for schedule(static) nowait
            for (std::size_t iInner = 0; iInner < nInner; iInner++)
            {
              std::size_t i1 = iInner * 8;
              kernel_core_unroll2(psi, i0+i1+0, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
              //kernel_core(psi, i0+i1+1, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
              kernel_core_unroll2(psi, i0+i1+2, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
              //kernel_core(psi, i0+i1+3, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
              kernel_core_unroll2(psi, i0+i1+4, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
              //kernel_core(psi, i0+i1+5, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
              kernel_core_unroll2(psi, i0+i1+6, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
              //kernel_core(psi, i0+i1+7, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            }
          }
        }
        else // nOuter > nInner
        {
          // single bit set, so we have only half of the iterations
#pragma omp for schedule(static)
          for (std::size_t iOuter = 0; iOuter < nOuter/2; iOuter++)
          {
            // reconstruct i0
            const auto tmp = iOuter*2*d0;
            std::size_t i0 = tmp % ctrlmask + ctrlmask + 2 * (tmp - (tmp%ctrlmask));

            for (std::size_t iInner = 0; iInner < nInner; iInner++)
            {
              std::size_t i1 = iInner * 8;
              kernel_core_unroll2(psi, i0+i1+0, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
              //kernel_core(psi, i0+i1+1, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
              kernel_core_unroll2(psi, i0+i1+2, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
              //kernel_core(psi, i0+i1+3, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
              kernel_core_unroll2(psi, i0+i1+4, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
              //kernel_core(psi, i0+i1+5, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
              kernel_core_unroll2(psi, i0+i1+6, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
              //kernel_core(psi, i0+i1+7, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            }
          }
        }
      }
      else // generic variant
      {
#pragma omp for collapse(2) schedule(static)
        for (std::size_t iOuter = 0; iOuter < nOuter; iOuter++)
          for (std::size_t iInner = 0; iInner < nInner; iInner++)
          {
            std::size_t i0 = iOuter * 2*d0;
            std::size_t i1 = iInner * 8;
            if ( ((i0+i1+0) & ctrlmask) == ctrlmask) kernel_core(psi, i0+i1+0, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            if ( ((i0+i1+1) & ctrlmask) == ctrlmask) kernel_core(psi, i0+i1+1, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            if ( ((i0+i1+2) & ctrlmask) == ctrlmask) kernel_core(psi, i0+i1+2, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            if ( ((i0+i1+3) & ctrlmask) == ctrlmask) kernel_core(psi, i0+i1+3, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            if ( ((i0+i1+4) & ctrlmask) == ctrlmask) kernel_core(psi, i0+i1+4, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            if ( ((i0+i1+5) & ctrlmask) == ctrlmask) kernel_core(psi, i0+i1+5, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            if ( ((i0+i1+6) & ctrlmask) == ctrlmask) kernel_core(psi, i0+i1+6, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
            if ( ((i0+i1+7) & ctrlmask) == ctrlmask) kernel_core(psi, i0+i1+7, d0, tmp_m0, tmp_m1, tmp_m2, tmp_m3);
          }
      }
    }
  }
}

