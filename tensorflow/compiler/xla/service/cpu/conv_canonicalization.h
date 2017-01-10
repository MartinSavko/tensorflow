/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CONV_CANONICALIZATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CONV_CANONICALIZATION_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass.h"

namespace xla {
namespace cpu {

// An HLO pass that canonicalizes the dimension numbers of all top-level
// convolutions in the given module.
//
// In order to hit the fast path of using Eigen's convolution implementation, a
// convolution's dimension numbers need to satisfy certain constraints (so
// called canonical convolutions). This pass expands non-canonical convolutions
// into reshapes and canonical convolutions, so that these non-canonical
// convolutions can run faster.
class ConvCanonicalization : public HloPass {
 public:
  ConvCanonicalization() : HloPass("convolution-canonicalization") {}
  ~ConvCanonicalization() override {}

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CONV_CANONICALIZATION_H_