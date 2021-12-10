#include "nn/msg_pass.h"

namespace gnn
{

template<typename Dtype>
void BindWeight(SpTensor<CPU, Dtype>*& target, SpTensor<CPU, Dtype>& output)
{
	target = &output;
}

template<typename Dtype>
void BindWeight(SpTensor<CPU, Dtype>*& target, SpTensor<GPU, Dtype>& output)
{
	if (target == nullptr)
		target = new SpTensor<CPU, Dtype>();
}

template<typename mode, typename Dtype>
IMsgPass<mode, Dtype>::IMsgPass(std::string _name, bool _average) : Factor(_name, PropErr::N), cpu_weight(nullptr), average(_average)
{

}

template<typename mode, typename Dtype>
void IMsgPass<mode, Dtype>::Forward(std::vector< std::shared_ptr<Variable> >& operands, 
									std::vector< std::shared_ptr<Variable> >& outputs, 
									Phase phase) 
{
	//ASSERT(operands.size() == 1, "unexpected input size for " << StrType());
	//ASSERT(outputs.size() == 1, "unexpected output size for " << StrType()); 

	//std::cout << "IMsgPass forwarding..." << std::endl;
	//std::cout << typeid(*operands[0].get()).name() << std::endl;

	//if (dynamic_cast<HyperGraphVar*>(operands[0].get())) {
	//	std::cout << "Dynamic cast success" << std::endl;
	//}

	auto* input_graph = dynamic_cast< HyperGraphVar* >(operands[0].get());

	//std::cout << "NumVars=" << input_graph->graph->num_nodes <<", NumEdges="
	//	<< input_graph->graph->num_edges << std::endl;

	auto& output = dynamic_cast<SpTensorVar<mode, Dtype>*>(outputs[0].get())->value;
	BindWeight(cpu_weight, output);
	//std::cout << "Initing CPU Weight..." << std::endl;
	InitCPUWeight(input_graph->graph);
	//std::cout << "Init CPU Weight done." << std::endl;
	if (mode::type == MatMode::gpu)
		output.CopyFrom(*(this->cpu_weight));

	//std::cout << "IMsgPass forward done." << std::endl;
}

INSTANTIATE_CLASS(IMsgPass)

//====================== Node2NodeMsgPass ========================
template<typename mode, typename Dtype>
void Node2NodeMsgPass<mode, Dtype>::InitCPUWeight(GraphStruct* graph)
{
    this->cpu_weight->Reshape({graph->num_nodes, graph->num_nodes});
	this->cpu_weight->ResizeSp(graph->num_edges, graph->num_nodes + 1);
	
	int nnz = 0;
	auto& data = this->cpu_weight->data;
	for (uint i = 0; i < graph->num_nodes; ++i)
	{
		data->row_ptr[i] = nnz;
		auto& list = graph->in_edges->head[i];
		for (size_t j = 0; j < list.size(); ++j)
		{
			data->val[nnz] = this->average ? 1.0 / list.size() : 1.0;
			data->col_idx[nnz] = list[j].second;
			nnz++;
		}
	}
	assert(nnz == (int)graph->num_edges);
	data->row_ptr[graph->num_nodes] = nnz;
}

template<typename mode, typename Dtype>
void Node2NodeMsgPass<mode, Dtype>::InitCPUWeight(HyperGraphStruct* graph)
{
	this->cpu_weight->Reshape({ graph->num_nodes, graph->num_nodes });
	int nnz_init = 0;	//<TODO>: add nnz_init as an attribute to HyperGraphStruct
	for (int i = 0; i < graph->num_nodes; i++) {
		nnz_init += graph->neighbors[i].size();
	}
	this->cpu_weight->ResizeSp(nnz_init, graph->num_nodes + 1);

	int nnz = 0;
	auto& data = this->cpu_weight->data;
	for (uint i = 0; i < graph->num_nodes; ++i)
	{
		data->row_ptr[i] = nnz;
		for (int neighbor : graph->neighbors[i]) {
			data->val[nnz] = this->average ? 1.0 / graph->neighbors[i].size() : 1.0;
			data->col_idx[nnz] = neighbor;
			nnz++;
		}
	}
	//assert(nnz == (int)graph->num_edges);
	data->row_ptr[graph->num_nodes] = nnz;
	//std::cout << "N2N constructed. nnz="<< nnz <<", data->nnz=" << data->nnz << std::endl;
}

INSTANTIATE_CLASS(Node2NodeMsgPass)

//====================== Edge2NodeMsgPass ========================
template<typename mode, typename Dtype>
void Edge2NodeMsgPass<mode, Dtype>::InitCPUWeight(GraphStruct* graph)
{
	this->cpu_weight->Reshape({graph->num_nodes, graph->num_edges});
	this->cpu_weight->ResizeSp(graph->num_edges, graph->num_nodes + 1);
	
	int nnz = 0;
	auto& data = this->cpu_weight->data;		
	for (uint i = 0; i < graph->num_nodes; ++i)
	{
		data->row_ptr[i] = nnz;
		auto& list = graph->in_edges->head[i];
		for (size_t j = 0; j < list.size(); ++j)
		{
			data->val[nnz] = this->average ? 1.0 / list.size() : 1.0;
			data->col_idx[nnz] = list[j].first;
			nnz++;
		}
	}
	assert(nnz == (int)graph->num_edges);
	data->row_ptr[graph->num_nodes] = nnz;    
}

template<typename mode, typename Dtype>
void Edge2NodeMsgPass<mode, Dtype>::InitCPUWeight(HyperGraphStruct* graph)
{
	this->cpu_weight->Reshape({ graph->num_nodes, graph->num_edges });
	int nnz_init = 0;
	for (int i = 0; i < graph->num_edges; i++) {
		nnz_init += graph->edge_list[i].size();
	}
	this->cpu_weight->ResizeSp(nnz_init, graph->num_nodes + 1);

	int nnz = 0;
	auto& data = this->cpu_weight->data;
	for (uint i = 0; i < graph->num_nodes; ++i)
	{
		data->row_ptr[i] = nnz;
		auto& list = graph->linked_edges->head[i];
		for (size_t j = 0; j < list.size(); ++j)
		{
			data->val[nnz] = this->average ? 1.0 / list.size() : 1.0;
			data->col_idx[nnz] = list[j];
			nnz++;
		}
	}
	//assert(nnz == (int)graph->num_edges);
	data->row_ptr[graph->num_nodes] = nnz;
	//std::cout << "E2N constructed. nnz=" << nnz <<", data->nnz=" << data->nnz << std::endl;
}

INSTANTIATE_CLASS(Edge2NodeMsgPass)

//====================== Node2EdgeMsgPass ========================
template<typename mode, typename Dtype>
void Node2EdgeMsgPass<mode, Dtype>::InitCPUWeight(GraphStruct* graph)
{
    int nnz = 0;
    this->cpu_weight->Reshape({graph->num_edges, graph->num_nodes});
    this->cpu_weight->ResizeSp(graph->num_edges, graph->num_edges + 1);
        
	auto& data = this->cpu_weight->data;
    for (uint i = 0; i < graph->num_edges; ++i)
    {
		data->row_ptr[i] = nnz;
		data->val[nnz] = 1.0;
		data->col_idx[nnz] = graph->edge_list[i].first;
		nnz++;
    }
    data->row_ptr[graph->num_edges] = nnz;
    assert(nnz == data->nnz);    
}

template<typename mode, typename Dtype>
void Node2EdgeMsgPass<mode, Dtype>::InitCPUWeight(HyperGraphStruct* graph)
{
	int nnz = 0;
	this->cpu_weight->Reshape({ graph->num_edges, graph->num_nodes });
	int nnz_init = 0;
	for (int i = 0; i < graph->num_edges; i++) {
		nnz_init += graph->edge_list[i].size();
	}
	this->cpu_weight->ResizeSp(nnz_init, graph->num_edges + 1);

	auto& data = this->cpu_weight->data;
	for (uint i = 0; i < graph->num_edges; ++i)
	{
		//<TODO>: check if the below code is correct
		data->row_ptr[i] = nnz;
		for (int nodeInEdge : graph->edge_list[i]) {
			data->val[nnz] = 1.0;
			data->col_idx[nnz] = nodeInEdge;
			nnz++;
		}
		/*std::vector<int> tmpEdge = graph->edge_list[i];
		for (int j = 0; j < tmpEdge.size(); j++) {
			data->val[nnz] = 1.0;
			data->col_idx[nnz] = tmpEdge[j];
			nnz++;
		}*/
	}
	data->row_ptr[graph->num_edges] = nnz;
	//assert(nnz == data->nnz);
	//std::cout << "N2E constructed. nnz="<<nnz<<", data->nnz="<<data->nnz<< std::endl;
}

INSTANTIATE_CLASS(Node2EdgeMsgPass)

//====================== Edge2EdgeMsgPass ========================
template<typename mode, typename Dtype>
void Edge2EdgeMsgPass<mode, Dtype>::InitCPUWeight(GraphStruct* graph)
{
    int nnz = 0;
    this->cpu_weight->Reshape({graph->num_edges, graph->num_edges});
    size_t cnt = 0;
    for (uint i = 0; i < graph->num_nodes; ++i)
    {
        auto in_cnt = graph->in_edges->head[i].size();
        cnt += in_cnt * (in_cnt - 1); 
    }
    this->cpu_weight->ResizeSp(cnt, graph->num_edges + 1);            
        
    auto& data = this->cpu_weight->data;
    for (uint i = 0; i < graph->num_edges; ++i)
    {
        data->row_ptr[i] = nnz;
        int node_from = graph->edge_list[i].first, node_to = graph->edge_list[i].second; 
        auto& list = graph->in_edges->head[node_from]; 
        for (size_t j = 0; j < list.size(); ++j)
        {
            if (list[j].second == node_to)
                continue; // the same edge in another direction
            data->val[nnz] = 1.0;
            data->col_idx[nnz] = list[j].first; // the edge index
            nnz++;
        }
    }
    data->row_ptr[graph->num_edges] = nnz;
    assert(nnz == data->nnz);
    assert(data->nnz == (int)cnt);
}

INSTANTIATE_CLASS(Edge2EdgeMsgPass)

//====================== SubgraphMsgPass ========================
template<typename mode, typename Dtype>
void SubgraphMsgPass<mode, Dtype>::InitCPUWeight(GraphStruct* graph)
{
	this->cpu_weight->Reshape({graph->num_subgraph, graph->num_nodes});		
	this->cpu_weight->ResizeSp(graph->num_nodes, graph->num_subgraph + 1);
	
	int nnz = 0;
	auto& data = this->cpu_weight->data;
	for (uint i = 0; i < graph->num_subgraph; ++i)
	{
		data->row_ptr[i] = nnz;
		auto& list = graph->subgraph->head[i];
		for (size_t j = 0; j < list.size(); ++j)
		{
			data->val[nnz] = this->average ? 1.0 / list.size() : 1.0;
			data->col_idx[nnz] = list[j];
			nnz++;
		}
	}
	assert(nnz == (int)graph->num_nodes);
	data->row_ptr[graph->num_subgraph] = nnz;
}

template<typename mode, typename Dtype>
void SubgraphMsgPass<mode, Dtype>::InitCPUWeight(HyperGraphStruct* graph)
{
	this->cpu_weight->Reshape({ graph->num_subgraph, graph->num_nodes });
	this->cpu_weight->ResizeSp(graph->num_nodes, graph->num_subgraph + 1);

	int nnz = 0;
	auto& data = this->cpu_weight->data;
	for (uint i = 0; i < graph->num_subgraph; ++i)
	{
		data->row_ptr[i] = nnz;
		auto& list = graph->subgraph->head[i];
		for (size_t j = 0; j < list.size(); ++j)
		{
			data->val[nnz] = this->average ? 1.0 / list.size() : 1.0;
			data->col_idx[nnz] = list[j];
			nnz++;
		}
	}
	assert(nnz == (int)graph->num_nodes);
	data->row_ptr[graph->num_subgraph] = nnz;
	//std::cout << "SubgraphMsgPass constructed. nnz=" << nnz << "data->nnz=" << data->nnz << std::endl;
}

INSTANTIATE_CLASS(SubgraphMsgPass)


//====================== SubgraphMsgPass_edge ========================
template<typename mode, typename Dtype>
void SubgraphMsgPass_edge<mode, Dtype>::InitCPUWeight(GraphStruct* graph)
{
	std::cout << "Error: SubgraphMsgPass_edge NOT IMPLEMENTED for GraphStruct!" << std::endl;
}

template<typename mode, typename Dtype>
void SubgraphMsgPass_edge<mode, Dtype>::InitCPUWeight(HyperGraphStruct* graph)
{
	this->cpu_weight->Reshape({ graph->num_subgraph, graph->num_edges});
	this->cpu_weight->ResizeSp(graph->num_edges, graph->num_subgraph + 1);

	int nnz = 0;
	auto& data = this->cpu_weight->data;
	for (uint i = 0; i < graph->num_subgraph; ++i)
	{
		data->row_ptr[i] = nnz;
		auto& list = graph->subgraph_edge->head[i];
		for (size_t j = 0; j < list.size(); ++j)
		{
			data->val[nnz] = this->average ? 1.0 / list.size() : 1.0;
			data->col_idx[nnz] = list[j];
			nnz++;
		}
	}
	assert(nnz == (int)graph->num_edges);
	data->row_ptr[graph->num_subgraph] = nnz;
	//std::cout << "SubgraphMsgPass constructed. nnz=" << nnz << "data->nnz=" << data->nnz << std::endl;
}


INSTANTIATE_CLASS(SubgraphMsgPass_edge)

}