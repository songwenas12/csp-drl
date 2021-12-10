#include "stdafx.h"
#include "nstep_replay_mem.h"
#include "i_env.h"
#include "config.h"
#include <cassert>

#define max(x, y) (x > y ? x : y)

namespace gnn {

	NStepReplayMem::NStepReplayMem() {

	}

	NStepReplayMem::NStepReplayMem(int _memory_size) {
		memory_size = _memory_size;
		constrGraphList.resize(memory_size);
		actionList.resize(memory_size);
		rewardList.resize(memory_size);
		sList.resize(memory_size);
		s_primeList.resize(memory_size);

		current = 0;
		count = 0;
		distribution = new std::uniform_int_distribution<int>(0, memory_size - 1);
	}

	NStepReplayMem::~NStepReplayMem() {
		delete this->distribution;
	}

	void NStepReplayMem::Clear() {
		current = count = 0;
	}

	void NStepReplayMem::Add(std::shared_ptr<operations_research::ConstrStruct> g,
		std::shared_ptr<operations_research::ConstrFeatures> s_t,
		int a_t,
		double r_t,
		std::shared_ptr<operations_research::ConstrFeatures> s_prime)
	{
		constrGraphList[current] = g;
		sList[current] = s_t;
		actionList[current] = a_t;
		rewardList[current] = r_t;
		s_primeList[current] = s_prime;

		count = max(count, current + 1);
		current = (current + 1) % memory_size;

	}

	void NStepReplayMem::Sampling(int batch_size, ReplaySample& result) {
		result.g_list.resize(batch_size);
		result.list_st.resize(batch_size);
		result.list_at.resize(batch_size);
		result.list_rt.resize(batch_size);
		result.list_s_primes.resize(batch_size);
		auto& dist = *distribution;
		for (int i = 0; i < batch_size; ++i) {
			int idx = dist(generator) % count;
			result.g_list[i] = constrGraphList[idx];
			//result.list_ConstrGraphID[i] = constrGraphIDList[idx];
			result.list_st[i] = sList[idx];
			result.list_at[i] = actionList[idx];
			result.list_rt[i] = rewardList[idx];
			result.list_s_primes[i] = s_primeList[idx];
		}
	}


	PEReplayMem::PEReplayMem(int capacity, int batch, double alpha, double beta, double eps)
		:tree_(capacity)
		, batch_(batch)
		, engine_()
		, dist_(0.0, 1.0){

		constrGraphList.resize(capacity);
		actionList.resize(capacity);
		rewardList.resize(capacity);
		sList.resize(capacity);
		s_primeList.resize(capacity);

		this->alpha = alpha;
		this->beta = beta;
		this->eps = eps;

		this->_indices.resize(batch);
		this->_priorities.resize(batch);
	}

	void PEReplayMem::Add(std::shared_ptr<operations_research::ConstrStruct> g, 
		std::shared_ptr<operations_research::ConstrFeatures> s_t, int a_t, 
		double r_t, std::shared_ptr<operations_research::ConstrFeatures> s_prime){
		
		int index = this->add_internal(this->max_priority); //already with _max_priority**alpha, see update_priorities
		constrGraphList[index] = g;
		sList[index] = s_t;
		actionList[index] = a_t;
		rewardList[index] = r_t;
		s_primeList[index] = s_prime;
	}

	bool PEReplayMem::Sampling(ReplaySample& result, std::vector<int>& indices, 
		std::vector<double>& weights, std::vector<double>& priorities){

		if (this->tree_.get_size() <= this->batch_size()) {
			std::cout << "Error in PEReplayMem [Memory size < batch_size]" << std::endl;
			return false;
		}

		this->sample_internal(this->_indices, this->_priorities);
		
		std::vector<int> ind;
		for (int i = 0; i < this->_indices.size(); i++) {
			if (this->_indices[i] >= 0)
				ind.push_back(i);
		}
		if (ind.size() <= 0) {
			//std::cout << "Error in PEReplayMem [Sample size <= 0]" << std::endl;
			return false;
		}

		// <TODO> check if it is needed to copy this->_indices to indices
		if (ind.size() != this->batch_size()) {
			std::cout << "It actually happens in PEReplayMem [Sample size != batch_size]" << std::endl;
		}
		indices.resize(ind.size());
		priorities.resize(ind.size());
		weights.resize(ind.size());
		for (int i = 0; i < ind.size(); i++) {
			indices[i] = this->_indices[ind[i]];
			priorities[i] = this->_priorities[ind[i]];
			weights[i] = std::pow(priorities[i], beta); //ignore product with N since we will normalize by max(weights)
		}
		double max_weight = weights[0];
		for (double weight_temp : weights) 
			max_weight = max(weight_temp, max_weight);

		for (int i = 0; i < weights.size(); i++) 
			weights[i] = weights[i] / (max_weight + this->eps);
		
		// Populate the ReplaySample
		int sample_size = indices.size();
		result.g_list.resize(sample_size);
		result.list_st.resize(sample_size);
		result.list_at.resize(sample_size);
		result.list_rt.resize(sample_size);
		result.list_s_primes.resize(sample_size);
		for (int i = 0; i < sample_size; ++i) {
			int idx = indices[i];
			result.g_list[i] = constrGraphList[idx];
			result.list_st[i] = sList[idx];
			result.list_at[i] = actionList[idx];
			result.list_rt[i] = rewardList[idx];
			result.list_s_primes[i] = s_primeList[idx];
		}
		return true;
	}

	void PEReplayMem::Update_Priorities(std::vector<int>& indices, 
		std::vector<double>& old_priorities, std::vector<double>& priorities){

		// Clip the priorities above eps, and then do the power
		for (int i = 0; i < priorities.size(); i++) {
			priorities[i] = max(priorities[i], this->eps);
			priorities[i] = std::pow(priorities[i], this->alpha);
		}

		// Update the max_priority
		double temp_max_priority = priorities[0];
		for (double priority : priorities)
			temp_max_priority = max(temp_max_priority, priority);
		this->max_priority = max(this->max_priority, temp_max_priority);

		for (int i = 0; i < old_priorities.size(); i++) {
			old_priorities[i] = old_priorities[i] * this->alpha;	// self._decay_alpha in the original code, 
																	// but seems not defined
		}
		for (int i = 0; i < priorities.size(); i++) {
			priorities[i] = max(priorities[i], old_priorities[i]);
		}
		this->update_priority_internal(indices, priorities);
	}


	int PEReplayMem::add_internal(double priority){
		return tree_.add(priority);
	}

	void PEReplayMem::sample_internal(std::vector<int>& indices, std::vector<double>& priorities){
		// MODIFICATION: here we directly operate on the input arrays (indices and priorities),
		// instead of wrap them using the cndarray class
		
		/*if (!(1 == indices_a.ndim() && 1 == priorities_a.ndim())) {
			throw std::runtime_error("Incorrect number of dimensions: indices, priorities.");
		}
		if (!(batch_ == indices_a.shape(0) && batch_ == priorities_a.shape(0))) {
			throw std::runtime_error("Incorrect number of shape: indices, priorities.");
		}*/
		const double num_els = tree_.get_size();
		if (num_els < batch_) {
			throw std::runtime_error("number elements < batch_size!");
		}

		//int_1darray indices(indices_a);
		//double_1darray priorities(priorities_a);

		///	const double capacity = tree_.capacity();

		///	double min_p = 10e10;
		for (int i = 0; i < batch_; ++i) {
			const double r = dist_(engine_);
			const std::pair<int, double> & res = tree_.find(r);
			//// index, priority = res.first, res.second
			if (res.first >= num_els) {
				//indices.ix(i) = -1;
				indices[i] = -1;
				continue;
			}

			/// const double w = res.second; /// > 1e-10 ? std::pow(res.second * capacity, -beta) : 1e-10;		
			///		if(res.second < min_p) min_p = res.second;

			//indices.ix(i) = res.first;
			//priorities.ix(i) = res.second;
			indices[i] = res.first;
			priorities[i] = res.second;

			tree_.update_value(res.first, 0);	///# To avoid duplicating
		}

		///	if(min_p <=0) min_p=1e-10;
		for (int i = 0; i < batch_; ++i) {
			/*if (indices.ix(i) >= 0) {
				tree_.update_value(indices.ix(i), priorities.ix(i));
			}*/
			if (indices[i] >= 0) {
				tree_.update_value(indices[i], priorities[i]);
			}
		}
		///	return min_p;
	}

	void PEReplayMem::update_priority_internal(const std::vector<int>& indices, const std::vector<double>& priorities){
		// MODIFICATION: same as PEReplayMem::sample()
		
		/*if (!(1 == indices_a.ndim() && 1 == priorities_a.ndim())) {
			throw std::runtime_error("Incorrect number of dimensions: indices, priorities");
		}

		if (!(indices_a.shape(0) == priorities_a.shape(0))) {
			throw std::runtime_error("Incorrect number of shape: indices, priorities");
		}*/

		//int_1darray indices(indices_a);
		//double_1darray priorities(priorities_a);

		/*for (int i = 0; i < indices.shape(0); ++i) {
			tree_.update_value(indices.ix(i), priorities.ix(i));
		}*/

		// <TODO>: check if use indeces.size() is correct and equal to the batch size 
		for (int i = 0; i < indices.size(); ++i) {
			tree_.update_value(indices[i], priorities[i]);
		}
	}

}