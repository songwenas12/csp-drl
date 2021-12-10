#ifndef Q_NET_H
#define Q_NET_H

#include "ortools/constraint_solver/constraint_solver.h"
#include "inet.h"
using namespace gnn;

class QNet : public INet
{
public:
    QNet();

    virtual void BuildNet() override;
    virtual void SetupTrain(std::vector<int>& idxes, 
		std::vector< std::shared_ptr<operations_research::ConstrStruct> >& g_list,
		std::vector< std::shared_ptr<operations_research::ConstrFeatures> >& list_s_primes,
                            std::vector<int>& actions, 
                            std::vector<double>& target) override;
                            
    virtual void SetupPredAll(std::vector<int>& idxes, 
		std::vector< std::shared_ptr<operations_research::ConstrStruct> >& g_list,
		std::vector< std::shared_ptr<operations_research::ConstrFeatures> >& list_s) override;

    void SetupGraphInput(std::vector<int>& idxes, 
		std::vector< std::shared_ptr<operations_research::ConstrStruct> >& g_list,
		std::vector< std::shared_ptr<operations_research::ConstrFeatures> >& list_s,
                         const int* actions);

    SpTensor<CPU, Dtype> act_select, rep_global;
    SpTensor<mode, Dtype> m_act_select, m_rep_global;

    //std::vector< std::set<int> > list_set;
};

#endif