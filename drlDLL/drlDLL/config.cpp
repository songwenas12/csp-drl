#pragma once
#include "stdafx.h"
#include "config.h"

int cfg::max_bp_iter = 1;
int cfg::embed_dim = 64;
int cfg::dev_id = 0;
int cfg::batch_size = 32;
int cfg::max_iter = 1;
int cfg::reg_hidden = 32;
int cfg::mlp_layers = 2;
int cfg::node_dim = 3;
int cfg::aux_dim = 0;
int cfg::min_n = 0;
int cfg::max_n = 0;
int cfg::mem_size = 0;
int cfg::n_step = -1;
int cfg::knn = -1;
int cfg::edge_dim = 1;
int cfg::edge_embed_dim = -1;
Dtype cfg::learning_rate = 0.0005;
Dtype cfg::lr_decay = 0.95;
Dtype cfg::decay = 0.9;
Dtype cfg::l2_penalty = 0;
Dtype cfg::momentum = 0;
Dtype cfg::w_scale = 0.01;
const char* cfg::net_type = "QNet";
int cfg::training_alg = 0;	//Using DQN by default
double cfg::eps_start = 1.0;
double cfg::eps_end = 0.05;
double cfg::eps_step = 1000.0;
int cfg::lr_step = 1000;
int cfg::max_infer_depth = -1;
bool cfg::use_PER = false;
double cfg::PER_alpha = 0.6;
double cfg::PER_beta = 0.4;
double cfg::PER_beta_anneal_step = 50000.0;
