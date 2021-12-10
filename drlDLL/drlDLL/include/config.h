#ifndef cfg_H
#define cfg_H

#include <iostream>
#include <cstring>
#include <fstream>
#include <set>
#include <map>
#include "util/gnn_macros.h"

typedef float Dtype;

#define GPU_MODE
#define USE_GPU

#ifdef GPU_MODE
typedef gnn::GPU mode;
#else
typedef gnn::CPU mode;
#endif

struct cfg
{
	static int max_bp_iter;
	static int embed_dim;
	static int batch_size;
	static int max_iter;
	static int dev_id;
	static int max_n, min_n;
	static int n_step;
	static int mem_size;
	static int mem_demo_size;
	static double batch_demo_ratio;
	static int reg_hidden;
	static int mlp_layers;	//Number of layers in the MLP
	static int node_dim;
	static int knn;
	static int edge_dim;
	static int edge_embed_dim;
	static int aux_dim;
	static Dtype decay;
	static Dtype learning_rate;
	static Dtype lr_decay;
	static Dtype l2_penalty;
	static Dtype momentum;
	static Dtype w_scale;
	static const char *net_type;	//<TODO>: change the types to string
	static int training_alg;	//0 for DQN, 1 for DDQN
	static double eps_start;
	static double eps_end;
	static double eps_step;
	static int lr_step;
	static int max_infer_depth;
	static bool use_PER;	//Whether use Prioritized Experience Replay
	static double PER_alpha;
	static double PER_beta;
	static double PER_beta_anneal_step;
};

#endif
