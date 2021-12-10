#include "sum_tree_base.h"

#include <cmath>

inline int pow2(int n){ return 2 << (n - 1); }

SumTreeBase::SumTreeBase(int capacity)
    : capacity_(capacity)
    , tree_level_( int(std::ceil(std::log2(capacity + 1)) + 1) )
    , tree_size_( pow2(tree_level_) - 1 )  //2^n = 2 << (n-1)
    , size_(0)
    , cursor_(0)
    , tree_(tree_size_, 0)
{}

int SumTreeBase::add(double value){
    const int index = cursor_;
    cursor_ += 1;
    if(cursor_ >= capacity_) cursor_ = 0;
    if(size_ < capacity_) size_ += 1;
    
    update_value(index, value);
    return index;
}

void SumTreeBase::update_value(int index, double value){
    const int tree_index = pow2(tree_level_ - 1) - 1 + index;
    const double diff = value - tree_[tree_index];
    _reconstruct(tree_index, diff);
}

void SumTreeBase::_reconstruct(int tindex, double diff){
    tree_[tindex] += diff;
    if(tindex > 0){
        _reconstruct( (tindex-1)/2, diff);
    }
}


double SumTreeBase::get_value(int index)const{
    const int tree_index = pow2(tree_level_ - 1) - 1 + index;
    return tree_[tree_index];
}

std::pair<int, double> SumTreeBase::find(double value, bool norm)const{
    if(norm){
        value *= tree_[0];
    }
    return _find(value, 0);
}

std::pair<int, double> SumTreeBase::_find(double value, int index)const{
    const int tree_index = pow2(tree_level_ - 1) - 1;
    if(tree_index <= index){
        return std::pair<int, double>( index - tree_index, tree_[index] );
    }
    const int left = 2*index + 1;
    if(value <= tree_[left]){ 
        return _find(value, left);  ///search left
    }
    return _find(value - tree_[left], left + 1); ///search right
}



