#include "stdafx.h"
#include "qnet.h"
#include "graph.h"
#include "config.h"
#include <iomanip>

QNet::QNet() : INet()
{
    inputs["node_feat"] = &m_node_feat;
    inputs["edge_feat"] = &m_edge_feat;
    inputs["label"] = &m_y;
    inputs["graph"] = &graph;
    inputs["act_select"] = &m_act_select;
    inputs["rep_global"] = &m_rep_global;
}

//New QNet: based on edge and node embeddings (modified)
void QNet::BuildNet()
{
	auto graph = add_const< HyperGraphVar >(fg, "graph", true);
	auto action_select = add_const< SpTensorVar<mode, Dtype> >(fg, "act_select", true);
	auto rep_global = add_const< SpTensorVar<mode, Dtype> >(fg, "rep_global", true);

	auto n2esum_param = af<Node2EdgeMsgPass<mode, Dtype>>(fg, { graph });
	auto e2nsum_param = af< Edge2NodeMsgPass<mode, Dtype> >(fg, { graph });
	auto subgsum_param = af< SubgraphMsgPass<mode, Dtype> >(fg, { graph });
	
	//Input to latent
	auto w_n2l = add_diff<DTensorVar>(model, "input-node-to-latent", { cfg::node_dim, cfg::embed_dim });
	auto w_e2l = add_diff<DTensorVar>(model, "input-edge-to-latent", { cfg::edge_dim, cfg::embed_dim });
	w_n2l->value.SetRandN(0, cfg::w_scale);
	w_e2l->value.SetRandN(0, cfg::w_scale);
	fg.AddParam(w_n2l);
	fg.AddParam(w_e2l);

	//MLP parameters
	std::vector< std::shared_ptr< DTensorVar<mode, Dtype>>> linear_VarMLP;
	std::vector< std::shared_ptr< DTensorVar<mode, Dtype>>> linear_ConstrMLP;
	linear_VarMLP.push_back(add_diff<DTensorVar>(model, "varMLPInput", { 2 * cfg::embed_dim + cfg::edge_dim, cfg::reg_hidden }));
	linear_ConstrMLP.push_back(add_diff<DTensorVar>(model, "constrMLPInput", { 2 * cfg::embed_dim + cfg::node_dim, cfg::reg_hidden }));	
	for (int i = 0; i < (cfg::mlp_layers - 2); i++) {
		linear_VarMLP.push_back(add_diff<DTensorVar>(model, "varMLPHidden", { cfg::reg_hidden, cfg::reg_hidden }));
		linear_ConstrMLP.push_back(add_diff<DTensorVar>(model, "constrMLPHidden", { cfg::reg_hidden, cfg::reg_hidden }));
	}
	linear_VarMLP.push_back(add_diff<DTensorVar>(model, "varMLPOutput", { cfg::reg_hidden, cfg::embed_dim }));
	linear_ConstrMLP.push_back(add_diff<DTensorVar>(model, "constrMLPOutput", { cfg::reg_hidden, cfg::embed_dim }));

	for (auto temp_param : linear_VarMLP) {
		temp_param->value.SetRandN(0, cfg::w_scale);
		fg.AddParam(temp_param);
	}
	for (auto temp_param : linear_ConstrMLP) {
		temp_param->value.SetRandN(0, cfg::w_scale);
		fg.AddParam(temp_param);
	}

	std::vector< std::shared_ptr< DTensorVar<mode, Dtype>>> linear_Q;
	linear_Q.push_back(add_diff<DTensorVar>(model, "qMLPInput", { 2 * cfg::embed_dim, cfg::reg_hidden }));
	for (int i = 0; i < (cfg::mlp_layers - 2); i++) {
		linear_Q.push_back(add_diff<DTensorVar>(model, "qMLPHidden", { cfg::reg_hidden, cfg::reg_hidden }));
	}
	linear_Q.push_back(add_diff<DTensorVar>(model, "qMLPOutput", { cfg::reg_hidden, 1 }));
	for (auto temp_param : linear_Q) {
		temp_param->value.SetRandN(0, cfg::w_scale);
		fg.AddParam(temp_param);
	}

	auto node_input = add_const< DTensorVar<mode, Dtype> >(fg, "node_feat", true);
	auto edge_input = add_const< DTensorVar<mode, Dtype> >(fg, "edge_feat", true);
	auto label = add_const< DTensorVar<mode, Dtype> >(fg, "label", true);

	auto node_init = af<MatMul>(fg, { node_input, w_n2l });
	auto edge_init = af< MatMul >(fg, { edge_input, w_e2l });

	int lv = 0;
	cur_node_embed = node_init;
	auto cur_edge_embed = edge_init;
	while (lv < cfg::max_bp_iter)
	{
		//Feed the node embeddings into the varMLP
		auto var_sum = af<MatMul>(fg, { n2esum_param, cur_node_embed });
		auto var_constr_concat = af<ConcatCols>(fg, { var_sum, cur_edge_embed, edge_input });
		auto hidden_varMLP = af<MatMul>(fg, { var_constr_concat, linear_VarMLP[0] });
		for (int i = 1; i < (cfg::mlp_layers - 1); i++) {
			hidden_varMLP = af<MatMul>(fg, { hidden_varMLP, linear_VarMLP[i] });
			hidden_varMLP = af<ReLU>(fg, { hidden_varMLP });
		}
		cur_edge_embed = af<MatMul>(fg, { hidden_varMLP, linear_VarMLP[cfg::mlp_layers - 1] });
	
		//Feed the edge embeddings into the constrMLP
		auto edge_sum = af<MatMul>(fg, { e2nsum_param, cur_edge_embed });
		auto constr_var_concat = af<ConcatCols>(fg, { edge_sum, cur_node_embed, node_input });
		auto hidden_constrMLP = af<MatMul>(fg, { constr_var_concat, linear_ConstrMLP[0] });
		for (int i = 1; i < (cfg::mlp_layers - 1); i++) {
			hidden_constrMLP = af<MatMul>(fg, { hidden_constrMLP, linear_ConstrMLP[i] });
			hidden_constrMLP = af<ReLU>(fg, { hidden_constrMLP });
		}
		cur_node_embed = af<MatMul>(fg, { hidden_constrMLP, linear_ConstrMLP[cfg::mlp_layers - 1] });

		lv++;
	}

	auto y_potential = af<MatMul>(fg, { subgsum_param, cur_node_embed });
	//auto y_potential_edge = af<MatMul>(fg, { subgsum_edge_param, cur_edge_embed });

	// Q func given a
	auto action_embed = af<MatMul>(fg, { action_select, cur_node_embed });
	auto embed_s_a = af< ConcatCols >(fg, { action_embed, y_potential });

	auto hidden_qMLP = af<MatMul>(fg, { embed_s_a, linear_Q[0] });
	for (int i = 1; i < (cfg::mlp_layers - 1); i++) {
		hidden_qMLP = af<MatMul>(fg, { hidden_qMLP, linear_Q[i] });
		hidden_qMLP = af<ReLU>(fg, { hidden_qMLP });
	}
	q_pred = af<MatMul>(fg, { hidden_qMLP, linear_Q[cfg::mlp_layers - 1] });

	auto diff = af< SquareError >(fg, { q_pred, label });
	loss = af< ReduceMean >(fg, { diff });

	// q func on all a
	auto rep_y = af<MatMul>(fg, { rep_global, y_potential });
	//auto rep_y_edge = af<MatMul>(fg, { rep_global, y_potential_edge });

	auto embed_s_a_all = af< ConcatCols >(fg, { cur_node_embed, rep_y });

	hidden_qMLP = af<MatMul>(fg, { embed_s_a_all, linear_Q[0] });
	for (int i = 1; i < (cfg::mlp_layers - 1); i++) {
		hidden_qMLP = af<MatMul>(fg, { hidden_qMLP, linear_Q[i] });
		hidden_qMLP = af<ReLU>(fg, { hidden_qMLP });
	}
	q_on_all = af<MatMul>(fg, { hidden_qMLP, linear_Q[cfg::mlp_layers - 1] });

	std::cout << "Using new QNet." << std::endl;
}

void QNet::SetupGraphInput(std::vector<int>& idxes, 
		std::vector< std::shared_ptr<operations_research::ConstrStruct> >& g_list,
		std::vector< std::shared_ptr<operations_research::ConstrFeatures> >& list_s,
                           const int* actions)
{
	int node_cnt = 0, edge_cnt = 0;
	for (size_t i = 0; i < idxes.size(); ++i)
    {
        auto& g = g_list[idxes[i]];
        node_cnt += g->numVars;
        auto& cur_Features = list_s[idxes[i]];
		for (int i = 0; i < cur_Features->varSize.size(); i++) {
			if (cur_Features->varSize[i] == 0)
				std::cout << "Error: domain of var " << i << " is empty!" << std::endl;
		}
        edge_cnt += g->numConstrs;	//<TODO>: edge is undirected. Check if this is correct.
    }

    graph.Resize(idxes.size(), node_cnt);

    node_feat.Reshape({(size_t)node_cnt, (size_t)cfg::node_dim});
    node_feat.Fill(0.0);	//<TODO>: check if 1.0 or 0.0
    edge_feat.Reshape({(size_t)edge_cnt, (size_t)cfg::edge_dim});
    edge_feat.Fill(0.0);

    if (actions) {
        act_select.Reshape({idxes.size(), (size_t)node_cnt});
        act_select.ResizeSp(idxes.size(), idxes.size() + 1);
    } else {
        rep_global.Reshape({(size_t)node_cnt, idxes.size()});
        rep_global.ResizeSp(node_cnt, node_cnt + 1);
    }	

    node_cnt = 0;
    edge_cnt = 0;
    size_t edge_offset = 0;
    for (size_t i = 0; i < idxes.size(); ++i)
	{                
        auto& g = g_list[idxes[i]];
		auto& g_Features = list_s[idxes[i]];
      
		for (int j = 0; j < g_Features->varSize.size(); j++) {
			node_feat.data->ptr[cfg::node_dim * (node_cnt + j) + 0] = g_Features->varSize[j];
			node_feat.data->ptr[cfg::node_dim * (node_cnt + j) + 1] = g_Features->ifBound[j];
		}
		
        for (int j = 0; j < g->numVars; ++j)
        {
            int x = node_cnt + j;
            graph.AddNode(i, x);
			
            if (!actions)	//<TODO>: check what this is used for
            {
                rep_global.data->row_ptr[node_cnt + j] = node_cnt + j;
                rep_global.data->val[node_cnt + j] = 1.0;
                rep_global.data->col_idx[node_cnt + j] = i;
            }
        }

		for (int j=0; j < g->numConstrs; j++)
		{
			std::vector<int> edgeNodes;
			for (int k = 0; k < g->constrVarIndex[j].size(); k++) {
				edgeNodes.push_back(g->constrVarIndex[j][k] + node_cnt);
			}

			graph.AddEdge(i, edge_cnt, edgeNodes);
			auto* edge_ptr = edge_feat.data->ptr + edge_offset;
			edge_ptr[0] = g_Features->tightness[j];
			edge_ptr[1] = g_Features->num_unbounded[j];
			edge_ptr[2] = g_Features->domainProduct[j];
			
			edge_offset += cfg::edge_dim;
			edge_cnt++;
		}


        if (actions)
        {
            auto act = actions[idxes[i]];
            assert(act >= 0 && act < g->num_nodes);
            act_select.data->row_ptr[i] = i;
            act_select.data->val[i] = 1.0;
            act_select.data->col_idx[i] = node_cnt + act;
        }
        node_cnt += g->numVars;
	}

	if (actions)
    {
        act_select.data->row_ptr[idxes.size()] = idxes.size();
        m_act_select.CopyFrom(act_select);
    } else {
        rep_global.data->row_ptr[node_cnt] = node_cnt;
        m_rep_global.CopyFrom(rep_global);
    }

	m_node_feat.CopyFrom(node_feat);
    m_edge_feat.CopyFrom(edge_feat);

}

void QNet::SetupTrain(std::vector<int>& idxes, 
	std::vector< std::shared_ptr<operations_research::ConstrStruct> >& g_list,
	std::vector< std::shared_ptr<operations_research::ConstrFeatures> >& list_s,
                      std::vector<int>& actions, 
                      std::vector<double>& target)
{    
	SetupGraphInput(idxes, g_list, list_s, actions.data());

    y.Reshape({idxes.size(), (size_t)1});
    for (size_t i = 0; i < idxes.size(); ++i)
        y.data->ptr[i] = target[idxes[i]];
    m_y.CopyFrom(y);
}

void QNet::SetupPredAll(std::vector<int>& idxes, 
	std::vector< std::shared_ptr<operations_research::ConstrStruct> >& g_list,
	std::vector< std::shared_ptr<operations_research::ConstrFeatures> >& list_s)
{    
	for (int i = 0; i < idxes.size(); i++) {
		if (list_s[idxes[i]]->ifTerminal == true)
			std::cout << "Error [SetupPredAll], input state " << i << " is terminal." << std::endl;
	}
    SetupGraphInput(idxes, g_list, list_s, nullptr);
}