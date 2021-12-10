// drlDLL.cpp : Defines the exported functions for the DLL application.
//

//#include "stdafx.h"
#include "ortools\constraint_solver\drl_interface.h"
#include "nstep_replay_mem.h"
#include "qnet.h"
#include "config.h"
#include <fstream>
#include <cstring>
#include <iomanip>
#include <random>
#include <algorithm>


#define inf 2147483647/2

namespace operations_research {

	// This class exists during the whole training process, holds the global 
	// information such as graph pool
	// In python, create a new MasterDRL object, and use it as argument when needed.
	// When finished, remember to DELETE this object properly.
	// In C++, use raw pointer of this object as argument.
	class MasterDRL : public IMasterDRL {
	public:
		std::map<int, std::shared_ptr<ConstrStruct>> caseStructPool;	//<caseID, the corresponding constrStruct>
		std::shared_ptr<gnn::NStepReplayMem> nStepMem;
		std::shared_ptr<gnn::PEReplayMem> PERMem;
		std::shared_ptr<QNet> net;	//The neural network
		Solver* crtSolver;

		MasterDRL(char* filePath) {
			//Parse the config
			this->parseConfig(filePath);
			this->numIterations = 0;
			this->crtLearningRate = cfg::learning_rate;
			this->crtEps = cfg::eps_start;
			this->maxInferDepth = cfg::max_infer_depth;
			#ifdef GPU_MODE
			GpuHandle::Init(cfg::dev_id, 1);
			std::cout << "GPU Handle initiated." << std::endl;
			#endif
			//Initialize the replay memory
			if (cfg::use_PER) {
				this->PERMem = std::make_shared<gnn::PEReplayMem>(cfg::mem_size, cfg::batch_size, 
					cfg::PER_alpha, cfg::PER_beta);
				std::cout << "PERMem initialized. size=" << this->PERMem->get_size() << std::endl;
			}
			else {
				this->nStepMem = std::make_shared<gnn::NStepReplayMem>(cfg::mem_size);
				std::cout << "nStepMem initialized. size=" << this->nStepMem->constrGraphList.size() << std::endl;
			}
			//Construct the QNet
			this->net = std::make_shared<QNet>();
			this->net->BuildNet();
			std::cout << "QNet built." << std::endl;
			//Initialize the predVector
			this->predVectors.push_back(new std::vector<double>(cfg::max_n));
			std::cout << "MasterDRL is succesfully constructed." << std::endl;
		}

		~MasterDRL() {
			for (int i = 0; i < this->predVectors.size(); i++) {
				delete this->predVectors[i];
			}
			#ifdef GPU_MODE
			GpuHandle::Destroy();
			#endif
		}

		void destroy() {
			delete this;
			std::cout << "MasterDRL is destroyed." << std::endl;
		}
		
		// Parse the config specified in the config file
		// <TODO>: remove non-useful parameters
		static void parseConfig(char* filePath) {
			//std::cout << "Loading config..." << filePath << std::endl;
			std::ifstream infile(filePath);
			std::string line;
			while (std::getline(infile, line)) {
				std::istringstream iss(line);
				std::string paramName;
				std::string paramValue;
				if (!(iss >> paramName >> paramValue)) { 
					std::cout << "Error in parsing" << std::endl;
					break; 
				}
				else {
					if (paramName.compare("learning_rate") == 0)
						cfg::learning_rate = atof(paramValue.c_str());
					if (paramName.compare("decay") == 0)
						cfg::decay = atof(paramValue.c_str());
					if (paramName.compare("max_bp_iter") == 0)
						cfg::max_bp_iter = atoi(paramValue.c_str());
					if (paramName.compare("dev_id") == 0)
						cfg::dev_id = atoi(paramValue.c_str());
					if (paramName.compare("embed_dim") == 0)
						cfg::embed_dim = atoi(paramValue.c_str());
					if (paramName.compare("node_dim") == 0)
						cfg::node_dim = atoi(paramValue.c_str());
					if (paramName.compare("edge_dim") == 0)
						cfg::edge_dim = atoi(paramValue.c_str());
					if (paramName.compare("knn") == 0)
						cfg::knn = atoi(paramValue.c_str());
					if (paramName.compare("edge_embed_dim") == 0)
						cfg::edge_embed_dim = atoi(paramValue.c_str());
					if (paramName.compare("reg_hidden") == 0)
						cfg::reg_hidden = atoi(paramValue.c_str());
					if (paramName.compare("mlp_layer") == 0)
						cfg::mlp_layers = atoi(paramValue.c_str());
					if (paramName.compare("max_n") == 0)
						cfg::max_n = atoi(paramValue.c_str());
					if (paramName.compare("min_n") == 0)
						cfg::min_n = atoi(paramValue.c_str());
					if (paramName.compare("mem_size") == 0)
						cfg::mem_size = atoi(paramValue.c_str());
					if (paramName.compare("n_step") == 0)
						cfg::n_step = atoi(paramValue.c_str());
					if (paramName.compare("batch_size") == 0)
						cfg::batch_size = atoi(paramValue.c_str());
					if (paramName.compare("max_iter") == 0)
						cfg::max_iter = atoi(paramValue.c_str());
					if (paramName.compare("l2") == 0)
						cfg::l2_penalty = atof(paramValue.c_str());
					if (paramName.compare("decay") == 0)
						cfg::decay = atof(paramValue.c_str());
					if (paramName.compare("w_scale") == 0)
						cfg::w_scale = atof(paramValue.c_str());
					if (paramName.compare("momentum") == 0)
						cfg::momentum = atof(paramValue.c_str());
					if (paramName.compare("net_type") == 0)
						cfg::net_type = paramValue.c_str();
					if (paramName.compare("training_alg") == 0)
						cfg::training_alg = atoi(paramValue.c_str());
					if (paramName.compare("eps_start") == 0)
						cfg::eps_start = atof(paramValue.c_str());
					if (paramName.compare("eps_end") == 0)
						cfg::eps_end = atof(paramValue.c_str());
					if (paramName.compare("eps_step") == 0)
						cfg::eps_step = atof(paramValue.c_str());
					if (paramName.compare("lr_step") == 0)
						cfg::lr_step = atoi(paramValue.c_str());
					if (paramName.compare("lr_decay") == 0)
						cfg::lr_decay = atof(paramValue.c_str());
					if (paramName.compare("max_infer_depth") == 0)
						cfg::max_infer_depth = atoi(paramValue.c_str());
					if (paramName.compare("use_PER") == 0) 
						cfg::use_PER = atoi(paramValue.c_str()) == 0 ? false : true;
					if (paramName.compare("PER_alpha") == 0)
						cfg::PER_alpha = atof(paramValue.c_str());
					if (paramName.compare("PER_beta") == 0)
						cfg::PER_beta = atof(paramValue.c_str());
					if (paramName.compare("PER_beta_anneal_step") == 0)
						cfg::PER_beta_anneal_step = atof(paramValue.c_str());
				}
			}
			std::cout << "Config loaded." << std::endl;
		}

		bool checkIfCaseInPool(int caseNumber) {
			if (this->caseStructPool.count(caseNumber)>0) {
				return true;
			}
			return false;
		}

		std::shared_ptr<ConstrStruct> retrieveConstrStruct(int caseNumber) {
			if (this->checkIfCaseInPool(caseNumber) == false) {
				std::cout << "Error: trying to retrive non-exist ConstrStruct!" << std::endl;
				return std::shared_ptr<ConstrStruct>(nullptr);
			}
			return this->caseStructPool.find(caseNumber)->second;
		}

		void addConstrStructToPool(int caseID, std::shared_ptr<ConstrStruct> constrStruct) {
			if (this->checkIfCaseInPool(caseID) == true) {
				std::cout << "Error: caseStruct already exists!" << std::endl;
				return;
			}
			this->caseStructPool[caseID] = constrStruct;
		}

		void setSolver(Solver* solver) {
			this->crtSolver = solver;
		}

		//Add a transaction to the replay memory
		void addTransactionToMemory(
			std::shared_ptr<operations_research::ConstrStruct>& g,
			std::shared_ptr<operations_research::ConstrFeatures>& s_t,
			int a_t,
			double r_t,
			std::shared_ptr<operations_research::ConstrFeatures>& s_prime, 
			bool ifLeftBranch
		) {
			if (cfg::use_PER) {
				this->PERMem->Add(g, s_t, a_t, r_t, s_prime);
				this->memoryCount = this->PERMem->get_size();
			}
			else {
				this->nStepMem->Add(g, s_t, a_t, r_t, s_prime);
				this->memoryCount = this->nStepMem->count;
			}
			if (ifLeftBranch == true)	// Only the decision on left branch is counted as a step
				this->UpdateParam();	//<TODO>: consider if we need to restore parameters after timeout
		}

		// For a list of states, use the QNet to predict the Q value of each legal action
		void Predict(std::vector< std::shared_ptr<operations_research::ConstrStruct> >& g_list,
			std::vector< std::shared_ptr<operations_research::ConstrFeatures> >& list_s,
			std::vector< std::vector<double>* >& pred) {
			
			//std::cout << "Predicting..." << std::endl;

			std::vector<int> batch_idxes;
			int n_graphs = g_list.size();
			DTensor<CPU, Dtype> output;
			for (int i = 0; i < n_graphs; i++) {
				if (list_s[i]->ifTerminal)
					continue;	//We do not need to predict terminal state
				batch_idxes.push_back(i);
			}

			net->SetupPredAll(batch_idxes, g_list, list_s);	//<TODO>: modify the SetupPredAll function for old_Qnet
			net->fg.FeedForward({ net->q_on_all }, net->inputs, Phase::TEST);

			auto& raw_output = net->q_on_all->value;
			output.CopyFrom(raw_output);
			int pos = 0;
			for (int i : batch_idxes) {
				auto& cur_pred = *(pred[i]);
				auto& g = g_list[i];
				//std::cout << "Predicted values are: ";
				for (int k = 0; k < g->numVars; ++k) {
					cur_pred[k] = output.data->ptr[pos];
					pos += 1;
					//std::cout << std::setprecision(5) << cur_pred[k] << " ";
				}

				auto tmp_state = list_s[i];
				for (int k = 0; k < g->numVars; k++) {
					if (tmp_state->varSize[k] == 1) {
						cur_pred[k] = inf;	//Set the value of illegal actions to inf
						//std::cout << "Set var " << k << " to inf=" << inf << std::endl;
					}
				}
			}
		}

		void PredictWithSnapshot(std::vector< std::shared_ptr<operations_research::ConstrStruct> >& g_list,
			std::vector< std::shared_ptr<operations_research::ConstrFeatures> >& list_s,
			std::vector< std::vector<double>* >& pred) {
			this->net->UseOldModel();
			Predict(g_list, list_s, pred);
			this->net->UseNewModel();
		}

		void TakeSnapshot() {
			this->net->old_model.DeepCopyFrom(net->model);
			std::cout << "Snapshot updated." << std::endl;
		}

		double Fit(const double lr) {
			double loss = 0.0;
			// Sampling
			ReplaySample sample;
			std::vector<int> indices;
			std::vector<double> weights, priorities, abs_td_error;
			//indices.resize(cfg::batch_size);
			//weights.resize(cfg::batch_size);
			//priorities.resize(cfg::batch_size);
			//td_error.resize(cfg::batch_size);
			if (cfg::use_PER) {
				bool PER_sample_flag = this->PERMem->Sampling(sample, indices, weights, priorities);
				if (PER_sample_flag == false)
					return 0.0;	//PERMem count is less than batch_size, hence do not learn
				abs_td_error.resize(indices.size());
			}
			else 
				this->nStepMem->Sampling(cfg::batch_size, sample);

			// Training
			switch (cfg::training_alg) {
			case 0:
				loss = Fit_DQN(lr, sample, abs_td_error);
			case 1:
				loss = Fit_DoubleDQN(lr, sample, abs_td_error);
			}
			// Update priorities for PER
			if (cfg::use_PER) 
				this->PERMem->Update_Priorities(indices, priorities, abs_td_error);
			return loss;
		}

		double Fit_DQN(const double lr, ReplaySample& sample, std::vector<double>& abs_td_error) {
			//std::cout << "Fitting [DQN]... lr=" << lr <<"batch_size="<<sample.g_list.size() << std::endl;
			//ReplaySample sample;
			std::vector<double> list_target;
			std::vector< std::vector<double>* > list_pred;
			list_pred.resize(cfg::batch_size);
			for (int i = 0; i < cfg::batch_size; ++i)
				list_pred[i] = new std::vector<double>(cfg::max_n + 10);
			//this->nStepMem->Sampling(cfg::batch_size, sample);

			this->PredictWithSnapshot(sample.g_list, sample.list_s_primes, list_pred);

			list_target.resize(cfg::batch_size);
			for (int i = 0; i < cfg::batch_size; ++i) {
				/*std::cout << "Computing target for sample " << i <<", ifTerminal="<< 
					sample.list_s_primes[i]->ifTerminal;*/
				double q_rhs = 0;	// y in Equation (6)
				if (sample.list_s_primes[i]->ifTerminal == false) {
					q_rhs = cfg::decay * this->min(sample.g_list[i]->numVars, list_pred[i]->data());
					//std::cout << " Predicted minQ=" << q_rhs;
					//if (q_rhs < 0)
					//	q_rhs = 0;
				}
				//std::cout << " rt=" << sample.list_rt[i];
				q_rhs += sample.list_rt[i];
				list_target[i] = q_rhs;
				//std::cout << " target=" << std::setprecision(5) << list_target[i] <<std::endl;
			}

			/*std::cout << "Targets are: ";
			for (int i = 0; i < list_target.size(); i++) {
				std::cout << list_target[i] << " ";
			}
			std::cout << std::endl;*/

			//<TODO>: find a more efficient way to handle list_pred
			for (int i = 0; i < cfg::batch_size; i++) {
				delete list_pred[i];
			}

			return Fit(lr, sample.g_list, sample.list_st, sample.list_at, list_target, abs_td_error);
		}

		double Fit_DoubleDQN(const double lr, ReplaySample& sample, std::vector<double>& abs_td_error) {
			//std::cout << "Fitting [DDQN]... lr=" << lr << std::endl;
			//ReplaySample sample;
			std::vector<double> list_target;
			std::vector< std::vector<double>* > list_pred;
			list_pred.resize(cfg::batch_size);
			for (int i = 0; i < cfg::batch_size; ++i)
				list_pred[i] = new std::vector<double>(cfg::max_n + 10);
			//this->nStepMem->Sampling(cfg::batch_size, sample);

			//For each sample, find the a*=argmin Q(S_{t+1})
			this->Predict(sample.g_list, sample.list_s_primes, list_pred);
			std::vector<int> predActions;
			predActions.resize(cfg::batch_size);
			for (int i = 0; i < cfg::batch_size; i++) {
				if (sample.list_s_primes[i]->ifTerminal)
					continue;	//Skip the terminal S_{t+1}
				predActions[i] = this->argmin(sample.g_list[i]->numVars, list_pred[i]->data());
			}
			
			//For each sample, use the oldModel to compute Q(S_{t+1}, a*)
			std::vector<double> QValue_old;
			QValue_old.resize(cfg::batch_size);
			DTensor<CPU, Dtype> output;
			this->net->UseOldModel();	//Switch to the target model
			std::vector<int> batch_idxes;
			for (int i = 0; i < cfg::batch_size; i++) {
				if (sample.list_s_primes[i]->ifTerminal)
					continue;	//Skip the terminal S_{t+1}
				batch_idxes.push_back(i);
			}
			this->net->SetupGraphInput(batch_idxes, sample.g_list, sample.list_s_primes, predActions.data());
			this->net->fg.FeedForward({ net->q_pred }, net->inputs, Phase::TEST);
			auto& raw_output = net->q_pred->value;
			output.CopyFrom(raw_output);
			int pos = 0;
			for (int i : batch_idxes) {
				QValue_old[i] = output.data->ptr[pos];
				pos++;
			}
			this->net->UseNewModel();	//Switch back to the online model
			
			//For each sample, compute the target value
			list_target.resize(cfg::batch_size);
			for (int i = 0; i < cfg::batch_size; ++i) {
				/*std::cout << "Computing target for sample " << i <<", ifTerminal="<<
					sample.list_s_primes[i]->ifTerminal;*/
				double q_rhs = 0;	// y in Equation (6)
				if (sample.list_s_primes[i]->ifTerminal == false) {
					q_rhs = cfg::decay * QValue_old[i];
					//if (q_rhs < 0)
					//	q_rhs = 0;
				}
				//std::cout << " rt=" << sample.list_rt[i];
				q_rhs += sample.list_rt[i];
				list_target[i] = q_rhs;
				//std::cout << " target=" << std::setprecision(5) << list_target[i] <<std::endl;
			}

			/*std::cout << "Targets are: ";
			for (int i = 0; i < list_target.size(); i++) {
				std::cout << list_target[i] << " ";
			}
			std::cout << std::endl;*/

			//<TODO>: find a more efficient way to handle list_pred
			for (int i = 0; i < cfg::batch_size; i++) {
				delete list_pred[i];
			}

			return Fit(lr, sample.g_list, sample.list_st, sample.list_at, list_target, abs_td_error);
		}

		double Fit(const double lr, std::vector< std::shared_ptr<operations_research::ConstrStruct> >& g_list,
			std::vector< std::shared_ptr<operations_research::ConstrFeatures> >& list_s,
			std::vector<int>& actions, 
			std::vector<double>& target,
			std::vector<double>& abs_td_error // For PER update
			) {
			//std::cout << "Real fitting..." << std::endl;
			std::vector<int> batch_idxes;
			Dtype loss = 0;
			int n_graphs = g_list.size();
			
			for (int i = 0; i < n_graphs; i++)
				batch_idxes.push_back(i);
			net->SetupTrain(batch_idxes, g_list, list_s, actions, target);	//<TODO>: modify the SetupTrain function for QNet and old_Qnet
			//std::cout << "SetupTrain done." << std::endl;
			net->fg.FeedForward({ net->loss }, net->inputs, Phase::TRAIN);
			//std::cout << "Feeding forward done. [Fit]" << std::endl;
			
			if (cfg::use_PER) {
				//Retrieve TD-error for PER
				//std::cout << "Retriving TD-error" << std::endl;
				auto& raw_output = net->q_pred->value;
				DTensor<CPU, Dtype> output;
				output.CopyFrom(raw_output);
				//std::cout << "Output copy done." << std::endl;
				for (int i = 0; i < batch_idxes.size(); i++) {
					abs_td_error[i] = std::abs(output.data->ptr[i] - target[i]);
				}

				/*std::cout << "TD-error retrieved." << std::endl;
				for (int i = 0; i < batch_idxes.size(); i++) {
					std::cout << abs_td_error[i] << " ";
				}
				std::cout << std::endl;*/
			}

			/*std::cout << "Predicted Q values: ";
			for (int i = 0; i < batch_idxes.size(); i++) {
				std::cout << output.data->ptr[i] <<" ";
			}
			std::cout << std::endl;*/
			
			net->fg.BackPropagate({ net->loss });
			//std::cout << "BP done." << std::endl;
			net->learner->cur_lr = lr;
			net->learner->Update();
			//std::cout << "Learner update done.";
			loss += net->loss->AsScalar() * n_graphs;

			//std::cout << "Training loss=" << loss/g_list.size() << std::endl;
			return loss / g_list.size();
		}

		double max(int n, const double* scores) {
			int pos = -1;
			double best = -10000000;
			for (int i = 0; i < n; ++i)
				if (pos == -1 || scores[i] > best) {
					pos = i;
					best = scores[i];
				}
			assert(pos >= 0);
			return best;
		}

		double min(int n, const double* scores) {
			int pos = 0;
			double best = scores[0];
			for (int i = 1; i < n; ++i)
				if (scores[i] < best) {
					pos = i;
					best = scores[i];
				}
			assert(pos >= 0);
			return best;
		}

		int argmin(int n, const double* scores) {
			int pos = 0;
			double best = scores[0];
			for (int i = 1; i < n; ++i)
				if (scores[i] < best) {
					pos = i;
					best = scores[i];
				}
			assert(pos >= 0);
			return pos;
		}

		void UpdateParam() {
			if (this->crtSolver->drl_phase != Solver::DRLPhase::WARM_UP) {
				this->numIterations++;	//Increace the iteration count by 1
				//Update Eps
				this->crtEps = cfg::eps_end + std::max(0., (cfg::eps_start - cfg::eps_end) *
					(cfg::eps_step - this->numIterations) / cfg::eps_step);
				//Update learning rate
				if (this->numIterations % cfg::lr_step == 0) {
					this->crtLearningRate = this->crtLearningRate*cfg::lr_decay;
				}
				//Update beta for PER
				if (cfg::use_PER) {
					if (this->PERMem->get_size() > cfg::batch_size) {
						//Only update beta when the size of PERMem reaches batch_size
						this->PERMem->beta = std::max(1.0, cfg::PER_beta + (1 - cfg::PER_beta) *
							(this->numIterations - cfg::batch_size) / cfg::PER_beta_anneal_step);
					}
				}
			}
		}

		void SaveModel(std::string filePath) {
			this->net->model.Save(filePath);
			std::cout << "Model saved." << std::endl;
		}

		void LoadModel(std::string filePath) {
			this->net->model.Load(filePath);
			std::cout << "Model loaded." << std::endl;
		}

		void PrintFeature(std::shared_ptr<operations_research::ConstrFeatures> cur_Features) {
			std::cout << "Feature information: related caseID=" << cur_Features->relateConstrStruct->caseID <<
				", ifTerminal=" << cur_Features->ifTerminal << std::endl;
			std::cout << "VarSize: ";
			for (int i = 0; i < cur_Features->varSize.size(); i++) {
				std::cout << i << "=" << cur_Features->varSize[i] << " ";
			}
			std::cout << std::endl;
			std::cout << "VarMin: ";
			for (int i = 0; i < cur_Features->varMin.size(); i++) {
				std::cout << i << "=" << cur_Features->varMin[i] << " ";
			}
			std::cout << std::endl;
			std::cout << "VarMax: ";
			for (int i = 0; i < cur_Features->varMax.size(); i++) {
				std::cout << i << "=" << cur_Features->varMax[i] << " ";
			}
			std::cout << std::endl;
			std::cout << "IfBound: ";
			for (int i = 0; i < cur_Features->ifBound.size(); i++) {
				std::cout << i << "=" << cur_Features->ifBound[i] << " ";
			}
			std::cout << std::endl;
			std::cout << "Tightness: ";
			for (int i = 0; i < cur_Features->tightness.size(); i++) {
				std::cout << i << "=" << std::setprecision(2) << cur_Features->tightness[i] << " ";
			}
			std::cout << std::endl;
		}

	};

	extern "C" {

		DECLDIR IMasterDRL* createMasterDRLFromDLL(char* filePath) {
			return new MasterDRL(filePath);
		}

	}
}