#pragma once

#define DECLDIR __declspec(dllexport)

#include "ortools/constraint_solver/constraint_solver.h"

namespace operations_research {
	
	/* The interface for the MasterDRL object
	*/
	class IMasterDRL {
	public:
		int maxInferDepth;	// The maximum search depth for DNN inference
		int numIterations;
		int memoryCount = 0;
		double crtLearningRate;
		double crtEps;
		std::vector< std::vector<double>* > predVectors;

		virtual void destroy() = 0;
		virtual bool checkIfCaseInPool(int caseNumber) = 0;
		virtual std::shared_ptr<ConstrStruct> retrieveConstrStruct(int caseNumber) = 0;
		virtual void addConstrStructToPool(int caseID, std::shared_ptr<ConstrStruct> constrStruct) = 0;
		virtual void setSolver(Solver* solver) = 0;
		virtual void addTransactionToMemory(
			std::shared_ptr<operations_research::ConstrStruct>& g,
			std::shared_ptr<operations_research::ConstrFeatures>& s_t,
			int a_t,
			double r_t,
			std::shared_ptr<operations_research::ConstrFeatures>& s_prime, 
			bool ifLeftBranch) = 0;
		virtual void Predict(std::vector< std::shared_ptr<operations_research::ConstrStruct> >& g_list,
			std::vector< std::shared_ptr<operations_research::ConstrFeatures> >& list_s,
			std::vector< std::vector<double>* >& pred) = 0;
		virtual void PredictWithSnapshot(std::vector< std::shared_ptr<operations_research::ConstrStruct> >& g_list,
			std::vector< std::shared_ptr<operations_research::ConstrFeatures> >& list_s,
			std::vector< std::vector<double>* >& pred) = 0;
		virtual double Fit(const double lr) = 0;
		virtual void TakeSnapshot() = 0;
		virtual void SaveModel(std::string modelFilePath) = 0;
		virtual void LoadModel(std::string modelFilePath) = 0;
	};

	extern "C"
	{
		DECLDIR IMasterDRL* createMasterDRLFromDLL(char* filePath);
	}
}