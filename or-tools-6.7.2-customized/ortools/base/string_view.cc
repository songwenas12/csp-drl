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

#include "ortools/base/string_view.h"
#include <iostream>  // NOLINT
#include <utility>

namespace absl {

std::ostream& operator<<(std::ostream& o, const string_view& piece) {
  o.write(piece.data(), piece.size());
  return o;
}

bool operator==(const string_view& x, const string_view& y) {
  int len = x.size();
  if (len != y.size()) {
    return false;
  }
  const char* p = x.data();
  const char* p2 = y.data();
  // Test last byte in case strings share large common prefix
  if ((len > 0) && (p[len - 1] != p2[len - 1])) return false;
  const char* p_limit = p + len;
  for (; p < p_limit; p++, p2++) {
    if (*p != *p2) return false;
  }
  return true;
}

bool operator<(const string_view& x, const string_view& y) {
  const int r = memcmp(x.data(), y.data(), std::min(x.size(), y.size()));
  return ((r < 0) || ((r == 0) && (x.size() < y.size())));
}

void string_view::CopyToString(std::string* target) const {
  target->assign(ptr_, length_);
}

int string_view::copy(char* buf, size_type n, size_type pos) const {
  int ret = std::min(length_ - pos, n);
  memcpy(buf, ptr_ + pos, ret);
  return ret;
}

int string_view::find(const string_view& s, size_type pos) const {
  if (length_ < 0 || pos > static_cast<size_type>(length_)) return npos;

  const char* result =
      std::search(ptr_ + pos, ptr_ + length_, s.ptr_, s.ptr_ + s.length_);
  const size_type xpos = result - ptr_;
  return xpos + s.length_ <= length_ ? xpos : npos;
}

int string_view::compare(const string_view& x) const {
  int r = memcmp(ptr_, x.ptr_, std::min(length_, x.length_));
  if (r == 0) {
    if (length_ < x.length_)
      r = -1;
    else if (length_ > x.length_)
      r = +1;
  }
  return r;
}

int string_view::find(char c, size_type pos) const {
  if (length_ <= 0 || pos >= static_cast<size_type>(length_)) {
    return npos;
  }
  const char* result = std::find(ptr_ + pos, ptr_ + length_, c);
  return result != ptr_ + length_ ? result - ptr_ : npos;
}

int string_view::rfind(const string_view& s, size_type pos) const {
  if (length_ < s.length_) return npos;
  const size_t ulen = length_;
  if (s.length_ == 0) return std::min(ulen, pos);

  const char* last = ptr_ + std::min(ulen - s.length_, pos) + s.length_;
  const char* result = std::find_end(ptr_, last, s.ptr_, s.ptr_ + s.length_);
  return result != last ? result - ptr_ : npos;
}

int string_view::rfind(char c, size_type pos) const {
  if (length_ <= 0) return npos;
  for (int i = std::min(pos, static_cast<size_type>(length_ - 1)); i >= 0;
       --i) {
    if (ptr_[i] == c) {
      return i;
    }
  }
  return npos;
}

string_view string_view::substr(size_type pos, size_type n) const {
  if (pos > length_) pos = length_;
  if (n > length_ - pos) n = length_ - pos;
  return string_view(ptr_ + pos, n);
}

const string_view::size_type string_view::npos = size_type(-1);

}  // namespace absl
