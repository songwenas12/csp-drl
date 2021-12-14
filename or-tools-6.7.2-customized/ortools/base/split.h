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

#ifndef OR_TOOLS_BASE_SPLIT_H_
#define OR_TOOLS_BASE_SPLIT_H_

#include <stddef.h>
#include <string>
#include <utility>
#include <vector>

#include "ortools/base/integral_types.h"
#include "ortools/base/logging.h"
#include "ortools/base/string_view.h"

namespace absl {
inline int SkipEmpty() { return 0xDEADBEEF; }

std::vector<std::string> StrSplit(const std::string& full, const char* delim, int flags);

std::vector<std::string> StrSplit(const std::string& full, char delim, int flags);

namespace delimiter {
inline const char* AnyOf(const char* x) { return x; }
}  // namespace delimiter
}  // namespace absl

// Split a std::string using a nul-terminated list of character
// delimiters.  For each component, parse using the provided
// parsing function and if successful, append it to 'result'.
// Return true if and only if all components parse successfully.
// If there are consecutive delimiters, this function skips over
// all of them.  This function will correctly handle parsing
// strings that have embedded \0s.
template <class T>
bool SplitStringAndParse(absl::string_view source, const std::string& delim,
                         bool (*parse)(const std::string& str, T* value),
                         std::vector<T>* result);

// We define here a very truncated version of the powerful absl::StrSplit()
// function. As of 2013-04, it can only be used like this:
// const char* separators = ...;
// std::vector<std::string> result = absl::StrSplit(
//    full, absl::delimiter::AnyOf(separators), absl::SkipEmpty());
//
// TODO(user): The current interface has a really bug prone side effect because
// it can also be used without the AnyOf(). If separators contains only one
// character, this is fine, but if it contains more, then the meaning is
// different: Split() should interpret the whole std::string as a delimiter. Fix
// this.
// ###################### TEMPLATE INSTANTIATIONS BELOW #######################
template <class T>
bool SplitStringAndParse(const std::string& source, const std::string& delim,
                         bool (*parse)(const std::string& str, T* value),
                         std::vector<T>* result) {
  CHECK(nullptr != parse);
  CHECK(nullptr != result);
  CHECK_GT(delim.size(), 0);
  const std::vector<std::string> pieces =
      ::absl::StrSplit(source, absl::delimiter::AnyOf(delim.c_str()),
                       static_cast<int64>(absl::SkipEmpty()));
  T t;
  for (absl::string_view piece : pieces) {
    if (!parse(piece.as_string(), &t)) return false;
    result->push_back(t);
  }
  return true;
}

#endif  // OR_TOOLS_BASE_SPLIT_H_
