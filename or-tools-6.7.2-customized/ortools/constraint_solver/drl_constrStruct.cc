#include "ortools/constraint_solver/constraint_solver.h"
#include "ortools/base/stl_util.h"
#include <iostream>
#include <iomanip>
#include <random>

using namespace operations_research;


//----- For class ConstrStruct ------
ConstrStruct::ConstrStruct(int caseID, std::vector<std::vector<int>> varConstrIndex,
	std::vector<std::vector<int>> constrVarIndex) {

	this->caseID = caseID;
	this->varConstrIndex = varConstrIndex;
	this->constrVarIndex = constrVarIndex;

	this->numVars = this->varConstrIndex.size();
	this->numConstrs = this->constrVarIndex.size();

	// Get the variable neighbors
	for (int i = 0; i < this->numVars; i++) {
		std::set<int> neighbor;
		for (int j : this->varConstrIndex[i]) {
			for (int neighbor_id : this->constrVarIndex[j]) {
				if (neighbor_id != i)
					neighbor.insert(neighbor_id);
			}
		}
		this->varNeighborIndex.push_back(neighbor);
	}
}

ConstrStruct::~ConstrStruct() {
	
}

void ConstrStruct::printVarSize(std::vector<int>& varSizeList) {
	std::cout << "----- VarSize: ";
	for (auto tmpSize : varSizeList) {
		std::cout << tmpSize << " ";
	}
	std::cout << std::endl;
}



//----- For class ConstrFeatures ------
ConstrFeatures::ConstrFeatures(Solver* relatedSolver, /*int sampleNum,*/ bool ifFail) {
	this->ifTerminal = true;
	this->relateConstrStruct = relatedSolver->instanceConstrStruct;
	this->varSize.resize(relatedSolver->varList.size());
	this->varMin.resize(relatedSolver->varList.size());
	this->varMax.resize(relatedSolver->varList.size());
	this->ifBound.resize(relatedSolver->varList.size());
	this->tightness.resize(relatedSolver->constrList.size());
	this->num_unbounded.resize(relatedSolver->constrList.size());
	this->domainProduct.resize(relatedSolver->constrList.size());

	if (ifFail == false) {
		//Get the variable features, except Ddeg
		for (int i = 0; i < relatedSolver->varList.size(); i++) {
			this->varSize[i] = relatedSolver->varList[i]->Size();
			if (this->varSize[i] > 1)
				this->ifTerminal = false;
			this->varMin[i] = relatedSolver->varList[i]->Min();
			this->varMax[i] = relatedSolver->varList[i]->Max();
			this->ifBound[i] = relatedSolver->varList[i]->Bound();
		}

		//Get the constraint features
		if (this->ifTerminal == false) {
			for (int i = 0; i < relatedSolver->constrList.size(); i++) {
				this->tightness[i] = ConstrFeatures::computeTableConstrTightness(relatedSolver, i);
				
				//Get the number of unbounded variables
				for (int j = 0; j < relatedSolver->constrList.size(); j++) {
					int tmpCount = 0;
					for (int i : relatedSolver->instanceConstrStruct->constrVarIndex[j]) {
						if (relatedSolver->varList[i]->Bound() == false)
							tmpCount++;
					}
					this->num_unbounded[j] = tmpCount;
				}
			}
		}
	}
}

ConstrFeatures::~ConstrFeatures() {
	
}

double ConstrFeatures::computeTableConstrTightness(Solver* relatedSolver, int constrID) {
	//Compute the product of domain sizes
	int domainProduct = 1;
	for (int j = 0; j < relatedSolver->instanceConstrStruct->constrVarIndex[constrID].size(); j++) {
		domainProduct *= relatedSolver->varList[relatedSolver->instanceConstrStruct->
			constrVarIndex[constrID][j]]->Size();
	}
	uint64 numActiveTuples = relatedSolver->constrList[constrID]->numActiveTuples();
	this->domainProduct[constrID] = domainProduct;
	return 1-(double)numActiveTuples / domainProduct;
}

void ConstrFeatures::printFeatures() {
	std::cout << "Feature information: related caseID="<<this->relateConstrStruct->caseID <<
		", ifTerminal="<< this->ifTerminal << std::endl;
	std::cout << "VarSize: ";
	for (int i = 0; i < this->varSize.size(); i++) {
		std::cout << i << "=" << this->varSize[i]<<" ";
	}
	std::cout << std::endl;
	std::cout << "VarMin: ";
	for (int i = 0; i < this->varMin.size(); i++) {
		std::cout << i << "=" << this->varMin[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "VarMax: ";
	for (int i = 0; i < this->varMax.size(); i++) {
		std::cout << i << "=" << this->varMax[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "IfBound: ";
	for (int i = 0; i < this->ifBound.size(); i++) {
		std::cout << i << "=" << this->ifBound[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "Tightness: ";
	for (int i = 0; i < this->tightness.size(); i++) {
		std::cout << i << "="<< std::setprecision(2) << this->tightness[i] << " ";
	}
	std::cout << std::endl;
}