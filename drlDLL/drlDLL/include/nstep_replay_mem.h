#ifndef NSTEP_REPLAY_MEM_H
#define NSTEP_REPLAY_MEM_H

//#include <vector>
#include <random>
#include "graph.h"
#include "i_env.h"
#include "ortools/constraint_solver/constraint_solver.h"

#include "sum_tree_base.h"
#include <random>
#include <cmath>

namespace gnn {

	class ReplaySample {
	public:
		std::vector< std::shared_ptr<operations_research::ConstrStruct> > g_list;
		//std::vector<int> list_ConstrGraphID;
		std::vector< std::shared_ptr<operations_research::ConstrFeatures> > list_st, list_s_primes;
		std::vector<int> list_at;
		std::vector<double> list_rt;

		//~ReplaySample() { std::cout << "ReplaySample destroyed." << std::endl; }
	};

	class NStepReplayMem {
	public:
		std::vector< std::shared_ptr<operations_research::ConstrStruct> > constrGraphList;
		std::vector<int> actionList;
		std::vector<double> rewardList;
		//A state (s or s_prime) is determined by two elements, the ConstrGraphID and the ConstrFeatures
		std::vector< std::shared_ptr<operations_research::ConstrFeatures> > sList, s_primeList;
		
		int current, count, memory_size;
		std::default_random_engine generator;
		std::uniform_int_distribution<int>* distribution;

		NStepReplayMem();
		NStepReplayMem(int memory_size);
		~NStepReplayMem();

		void Add(std::shared_ptr<operations_research::ConstrStruct> g,
			std::shared_ptr<operations_research::ConstrFeatures> s_t,
			int a_t,
			double r_t,
			std::shared_ptr<operations_research::ConstrFeatures> s_prime);

		void Sampling(int batch_size, ReplaySample& result);

		void Clear();
	};

	// Prioritized Experience Replay
	class PEReplayMem {
	public:
		PEReplayMem(int capacity, int batch, double alpha, double beta, double eps = 1e-10);

		void Add(std::shared_ptr<operations_research::ConstrStruct> g,
			std::shared_ptr<operations_research::ConstrFeatures> s_t, int a_t,
			double r_t, std::shared_ptr<operations_research::ConstrFeatures> s_prime);

		bool Sampling(ReplaySample& result, std::vector<int>& indices, 
			std::vector<double>& weights, std::vector<double>& priorities);

		void Update_Priorities(std::vector<int>& indices, std::vector<double>& old_priorities, 
			std::vector<double>& priorities);

		std::vector<int> _indices;
		std::vector<double> _priorities;

		// Parameters
		double alpha;
		double beta;
		double eps;
		double max_priority = 1.0;

		// Data
		std::vector< std::shared_ptr<operations_research::ConstrStruct> > constrGraphList;
		std::vector<int> actionList;
		std::vector<double> rewardList;
		std::vector< std::shared_ptr<operations_research::ConstrFeatures> > sList, s_primeList;
		
		inline int capacity() const { return tree_.capacity(); }
		inline int get_size()const { return tree_.get_size(); }
		inline int batch_size()const { return batch_; }
	protected:
		SumTreeBase tree_;
		const int batch_;

	private:
		std::mt19937 engine_;
		std::uniform_real_distribution<double> dist_;

		// Below functions should be internal
		int add_internal(double priority);

		void sample_internal(std::vector<int>& indices, std::vector<double>& priorities_a);

		void update_priority_internal(const std::vector<int> & indices, const std::vector<double> & priorities);


	};


}

#endif