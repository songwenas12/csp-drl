#ifndef SHARE_SUM_TREE_BASE_H
#define SHARE_SUM_TREE_BASE_H

#include <vector>
#include <utility>


class SumTreeBase{
public:
    SumTreeBase(int capacity);
    
    int add(double value);
    void update_value(int index, double value);

    std::pair<int, double> find(double value, bool norm=true)const;
    double get_value(int index)const;

    inline double tree_value(int tindex)const{ return tree_[tindex]; }
    inline int get_size()const{return size_;}
    inline int capacity() const {return capacity_; }
    inline int tree_level() const { return tree_level_;}
protected:
    const int capacity_;

private:
    const int tree_level_;
    const int tree_size_;
    int size_;
    int cursor_;
    std::vector<double> tree_;


    void _reconstruct(int tindex, double diff);

    std::pair<int, double> _find(double value, int index)const;
};

#endif
