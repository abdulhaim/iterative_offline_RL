#!/bin/bash

python eval_policy.py evaluator.generation_kwargs.adv_weight=0.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql_beta_0.pkl
python eval_policy.py evaluator.generation_kwargs.adv_weight=0.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql_beta_1.pkl
python eval_policy.py evaluator.generation_kwargs.adv_weight=0.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql_beta_2.pkl
python eval_policy.py evaluator.generation_kwargs.adv_weight=0.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql_beta_4.pkl
python eval_policy.py evaluator.generation_kwargs.adv_weight=0.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql_beta_8.pkl
python eval_policy.py evaluator.generation_kwargs.adv_weight=0.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql_beta_16.pkl
python eval_policy.py evaluator.generation_kwargs.adv_weight=0.0 eval.log_save_path=outputs/craigslist/eval_logs/seller_v_buyer_ilql_beta_32.pkl