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

inline void kernel_core(std::complex<double>* psi, std::size_t I, std::size_t d0, std::complex<double> m[2][2])
{
    std::complex<double> v[2];
    v[0] = psi[I];
    v[1] = psi[I + d0];

    psi[I] = (add(mul(v[0], m[0][0]), mul(v[1], m[0][1])));
    psi[I + d0] = (add(mul(v[0], m[1][0]), mul(v[1], m[1][1])));

}

// bit indices id[.] are given from high to low (e.g. control first for CNOT)
template <class V, class M>
void kernel(V &psi, unsigned id0, M const& m, std::size_t ctrlmask)
{
    const std::size_t n = psi.size();
    const std::size_t d0 = 1UL << id0;
    // let compiler know m and psi do not alias
    std::complex<double> tmpM[2][2];
    tmpM[0][0] = m[0][0];
    tmpM[1][0] = m[1][0];
    tmpM[0][1] = m[0][1];
    tmpM[1][1] = m[1][1];
    // calculate number of iterations
    const std::size_t nOuter = n / (2*d0);
    std::complex<double>* psiData = psi.data();

    if (ctrlmask == 0){
        #pragma omp parallel for schedule(static) if(n>1000)
        for (std::size_t iOuter = 0; iOuter < nOuter; iOuter++)
            for (std::size_t i1 = 0; i1 < d0; ++i1){
                std::size_t i0 = iOuter * 2*d0;
                kernel_core(psiData, i0 + i1, d0, tmpM);
            }
    }
    else{
        #pragma omp parallel for schedule(static) if(n>1000)
        for (std::size_t iOuter = 0; iOuter < nOuter; iOuter++){
            std::size_t i0 = iOuter * 2*d0;
            for (std::size_t i1 = 0; i1 < d0; ++i1){
                if (((i0 + i1)&ctrlmask) == ctrlmask)
                    kernel_core(psiData, i0 + i1, d0, tmpM);
            }
        }
    }
}

