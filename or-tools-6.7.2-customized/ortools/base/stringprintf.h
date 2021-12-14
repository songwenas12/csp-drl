// Copyright 2010-2017 Google
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OR_TOOLS_BASE_STRINGPRINTF_H_
#define OR_TOOLS_BASE_STRINGPRINTF_H_

#include <string>

namespace operations_research {
std::string StringPrintf(const char* const format, ...);
void SStringPrintf(std::string* const dst, const char* const format, ...);
void StringAppendF(std::string* const dst, const char* const format, ...);
}  // namespace operations_research

namespace absl {
std::string StrFormat(const char* const format, ...);
void StrAppendFormat(std::string* const dst, const char* const format, ...);
}  // namespace absl
#endif  // OR_TOOLS_BASE_STRINGPRINTF_H_
