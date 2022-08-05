#!/bin/bash

# IQL_tau=0.5, CQL_weight=0.1

# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.5_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=0.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.5_cql0.1_beta_0.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.5_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=1.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.5_cql0.1_beta_1.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.5_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=2.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.5_cql0.1_beta_2.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.5_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=4.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.5_cql0.1_beta_4.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.5_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=8.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.5_cql0.1_beta_8.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.5_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=16.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.5_cql0.1_beta_16.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.5_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=32.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.5_cql0.1_beta_32.pkl

# IQL_tau=0.7 CQL_weight=0.1

# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.7_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=0.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.7_cql0.1_beta_0.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.7_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=1.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.7_cql0.1_beta_1.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.7_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=2.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.7_cql0.1_beta_2.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.7_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=4.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.7_cql0.1_beta_4.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.7_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=8.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.7_cql0.1_beta_8.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.7_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=16.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.7_cql0.1_beta_16.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.7_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=32.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.7_cql0.1_beta_32.pkl

# IQL_tau=0.8 CQL_weight=0.1

# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.8_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=0.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.8_cql0.1_beta_0.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.8_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=1.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.8_cql0.1_beta_1.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.8_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=2.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.8_cql0.1_beta_2.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.8_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=4.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.8_cql0.1_beta_4.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.8_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=8.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.8_cql0.1_beta_8.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.8_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=16.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.8_cql0.1_beta_16.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.8_cql0.1_test2/model_311295.pkl evaluator.generation_kwargs.adv_weight=32.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.8_cql0.1_beta_32.pkl

# IQL_tau=0.9 CQL_weight=0.1

# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.9_cql0.1_test2/model_278527.pkl evaluator.generation_kwargs.adv_weight=0.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.9_cql0.1_beta_0.pkl
python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.9_cql0.1_test2/model_278527.pkl evaluator.generation_kwargs.adv_weight=1.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.9_cql0.1_beta_1.pkl
python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.9_cql0.1_test2/model_278527.pkl evaluator.generation_kwargs.adv_weight=2.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.9_cql0.1_beta_2.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.9_cql0.1_test2/model_278527.pkl evaluator.generation_kwargs.adv_weight=4.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.9_cql0.1_beta_4.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.9_cql0.1_test2/model_278527.pkl evaluator.generation_kwargs.adv_weight=8.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.9_cql0.1_beta_8.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.9_cql0.1_test2/model_278527.pkl evaluator.generation_kwargs.adv_weight=16.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.9_cql0.1_beta_16.pkl
# python eval_policy.py model.load.checkpoint_path=outputs/craigslist/craigslist_iql_tau0.9_cql0.1_test2/model_278527.pkl evaluator.generation_kwargs.adv_weight=32.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql2_tau0.9_cql0.1_beta_32.pkl

