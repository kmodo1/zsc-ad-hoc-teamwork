
## random3 / s2-mix / partners: biased (20)

| method | mean return | std across seeds | seeds |
|---|---:|---:|---:|
| oracle_best_sub_policy | 118.7 | 0.0 | 1 |
| static_ensemble_vote | 105.5 | 0.0 | 1 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | 105.2 | 2.2 | 3 |
| static_ensemble_argmax | 104.7 | 0.0 | 1 |
| best_fixed_sub_policy | 104.6 | 0.0 | 1 |
| actor_critic-sp+biased/top_learned_sub_policy | 102.9 | 1.4 | 3 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | 99.0 | 2.1 | 3 |
| actor_critic-sp+biased/policy_selection | 98.6 | 2.7 | 3 |
| actor_critic-sp+biased-memoryless/top_learned_sub_policy | 98.2 | 3.4 | 3 |
| actor_critic-sp+biased/adaptive_ensemble | 97.8 | 1.5 | 3 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | 92.0 | 1.1 | 3 |
| mep_s36 | 91.6 | 15.1 | 5 |
| actor_critic-sp+biased-memoryless/policy_selection | 89.6 | 0.8 | 3 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | 89.4 | 1.3 | 3 |
| actor_critic-sp+biased/learned_static_ensemble | 88.7 | 2.2 | 3 |
| fcp_s36 | 88.6 | 6.8 | 5 |
| fcp_s36_argmax | 88.3 | 5.0 | 5 |
| hsp_s36 | 87.2 | 5.7 | 5 |
| single_sub_policy | 86.1 | 13.6 | 12 |
| mep_s36_argmax | 85.9 | 21.9 | 5 |
| hsp_s36_argmax | 84.7 | 10.8 | 5 |
| static_ensemble | 82.5 | 0.0 | 1 |
| single_sub_policy_argmax | 81.5 | 18.0 | 12 |
| traj_s36 | 77.6 | 10.0 | 5 |
| e3t | 76.5 | 3.8 | 5 |
| cole_s75 | 75.1 | 12.1 | 5 |
| e3t_argmax | 73.7 | 3.9 | 5 |
| traj_s36_argmax | 73.6 | 14.0 | 5 |
| cole_s75_argmax | 71.3 | 19.1 | 5 |
| mappo | 20.6 | 5.2 | 5 |
| mappo_argmax | 18.1 | 5.5 | 5 |

## Differences on biased partners (ours - reference, 95% bootstrap CI over distinct partner agents)

| ours | reference | difference | 95% CI | partners better |
|---|---|---:|---|---:|
| actor_critic-sp+biased/adaptive_ensemble | actor_critic-sp+biased/learned_static_ensemble | +9.1 | [+3.9, +14.4] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble | actor_critic-sp+biased/top_learned_sub_policy | -5.2 | [-13.0, +2.9] | 7/20 |
| actor_critic-sp+biased/adaptive_ensemble | static_ensemble | +15.3 | [+9.5, +21.1] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble | static_ensemble_argmax | -6.9 | [-13.7, +1.0] | 7/20 |
| actor_critic-sp+biased/adaptive_ensemble | single_sub_policy | +11.7 | [+5.8, +17.2] | 16/20 |
| actor_critic-sp+biased/adaptive_ensemble | best_fixed_sub_policy | -6.8 | [-13.2, +0.4] | 5/20 |
| actor_critic-sp+biased/adaptive_ensemble | oracle_best_sub_policy | -20.9 | [-28.2, -13.8] | 0/20 |
| actor_critic-sp+biased/adaptive_ensemble | mappo | +77.2 | [+62.1, +91.5] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble | traj_s36 | +20.2 | [+11.3, +28.6] | 17/20 |
| actor_critic-sp+biased/adaptive_ensemble | hsp_s36 | +10.5 | [-1.1, +21.0] | 15/20 |
| actor_critic-sp+biased/adaptive_ensemble | fcp_s36 | +9.1 | [+2.5, +15.8] | 14/20 |
| actor_critic-sp+biased/adaptive_ensemble | mep_s36 | +6.2 | [-2.2, +15.4] | 12/20 |
| actor_critic-sp+biased/adaptive_ensemble | cole_s75 | +22.7 | [+13.0, +33.0] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble | e3t | +21.2 | [+12.2, +31.6] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble | mappo_argmax | +79.6 | [+64.5, +94.5] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble | traj_s36_argmax | +24.1 | [+13.2, +34.4] | 16/20 |
| actor_critic-sp+biased/adaptive_ensemble | hsp_s36_argmax | +13.0 | [+0.9, +25.1] | 14/20 |
| actor_critic-sp+biased/adaptive_ensemble | fcp_s36_argmax | +9.5 | [+1.3, +18.6] | 13/20 |
| actor_critic-sp+biased/adaptive_ensemble | mep_s36_argmax | +11.9 | [+3.4, +21.3] | 16/20 |
| actor_critic-sp+biased/adaptive_ensemble | cole_s75_argmax | +26.5 | [+16.5, +37.3] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble | e3t_argmax | +24.1 | [+13.9, +36.2] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble | best_baseline | +6.2 | [-2.2, +15.4] | 12/20 |
| actor_critic-sp+biased/adaptive_ensemble | best_baseline_argmax | +9.5 | [+1.3, +18.6] | 13/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | actor_critic-sp+biased/learned_static_ensemble | +16.6 | [+8.7, +23.7] | 17/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | actor_critic-sp+biased/top_learned_sub_policy | +2.3 | [-7.7, +11.5] | 12/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | static_ensemble | +22.8 | [+14.5, +30.3] | 17/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | static_ensemble_argmax | +0.5 | [-4.6, +5.8] | 11/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | single_sub_policy | +19.2 | [+11.5, +25.9] | 17/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | best_fixed_sub_policy | +0.7 | [-8.5, +9.4] | 13/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | oracle_best_sub_policy | -13.4 | [-23.4, -4.1] | 6/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mappo | +84.7 | [+67.5, +101.1] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | traj_s36 | +27.7 | [+18.1, +36.6] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | hsp_s36 | +18.0 | [+5.0, +29.6] | 16/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | fcp_s36 | +16.6 | [+9.5, +23.4] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mep_s36 | +13.7 | [+4.1, +24.1] | 16/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | cole_s75 | +30.2 | [+19.0, +41.5] | 17/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | e3t | +28.7 | [+19.8, +37.0] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mappo_argmax | +87.1 | [+69.6, +104.0] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | traj_s36_argmax | +31.6 | [+20.8, +41.5] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | hsp_s36_argmax | +20.5 | [+7.7, +32.6] | 17/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | fcp_s36_argmax | +16.9 | [+9.0, +24.3] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mep_s36_argmax | +19.4 | [+9.5, +29.9] | 17/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | cole_s75_argmax | +34.0 | [+23.0, +45.5] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | e3t_argmax | +31.6 | [+21.8, +41.4] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | best_baseline | +13.7 | [+4.1, +24.1] | 16/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | best_baseline_argmax | +16.9 | [+9.0, +24.3] | 18/20 |
| actor_critic-sp+biased/policy_selection | actor_critic-sp+biased/learned_static_ensemble | +10.0 | [+3.8, +15.9] | 15/20 |
| actor_critic-sp+biased/policy_selection | actor_critic-sp+biased/top_learned_sub_policy | -4.3 | [-12.0, +3.4] | 9/20 |
| actor_critic-sp+biased/policy_selection | static_ensemble | +16.1 | [+9.3, +22.8] | 18/20 |
| actor_critic-sp+biased/policy_selection | static_ensemble_argmax | -6.1 | [-12.5, +1.9] | 4/20 |
| actor_critic-sp+biased/policy_selection | single_sub_policy | +12.6 | [+6.3, +18.4] | 16/20 |
| actor_critic-sp+biased/policy_selection | best_fixed_sub_policy | -6.0 | [-12.4, +1.0] | 4/20 |
| actor_critic-sp+biased/policy_selection | oracle_best_sub_policy | -20.1 | [-27.6, -12.7] | 0/20 |
| actor_critic-sp+biased/policy_selection | mappo | +78.0 | [+62.2, +93.1] | 20/20 |
| actor_critic-sp+biased/policy_selection | traj_s36 | +21.1 | [+12.4, +29.6] | 19/20 |
| actor_critic-sp+biased/policy_selection | hsp_s36 | +11.4 | [-0.8, +22.3] | 15/20 |
| actor_critic-sp+biased/policy_selection | fcp_s36 | +10.0 | [+3.2, +16.9] | 15/20 |
| actor_critic-sp+biased/policy_selection | mep_s36 | +7.0 | [-1.4, +16.0] | 12/20 |
| actor_critic-sp+biased/policy_selection | cole_s75 | +23.6 | [+14.2, +33.3] | 18/20 |
| actor_critic-sp+biased/policy_selection | e3t | +22.1 | [+13.0, +32.5] | 18/20 |
| actor_critic-sp+biased/policy_selection | mappo_argmax | +80.5 | [+64.4, +96.0] | 20/20 |
| actor_critic-sp+biased/policy_selection | traj_s36_argmax | +25.0 | [+14.4, +35.4] | 18/20 |
| actor_critic-sp+biased/policy_selection | hsp_s36_argmax | +13.9 | [+1.3, +26.2] | 15/20 |
| actor_critic-sp+biased/policy_selection | fcp_s36_argmax | +10.3 | [+2.2, +19.6] | 14/20 |
| actor_critic-sp+biased/policy_selection | mep_s36_argmax | +12.7 | [+4.5, +21.7] | 16/20 |
| actor_critic-sp+biased/policy_selection | cole_s75_argmax | +27.3 | [+17.8, +37.4] | 19/20 |
| actor_critic-sp+biased/policy_selection | e3t_argmax | +24.9 | [+15.1, +36.9] | 19/20 |
| actor_critic-sp+biased/policy_selection | best_baseline | +7.0 | [-1.4, +16.0] | 12/20 |
| actor_critic-sp+biased/policy_selection | best_baseline_argmax | +10.3 | [+2.2, +19.6] | 14/20 |
| actor_critic-sp+biased/learned_static_ensemble | static_ensemble | +6.2 | [+4.2, +7.8] | 19/20 |
| actor_critic-sp+biased/learned_static_ensemble | static_ensemble_argmax | -16.0 | [-22.5, -8.9] | 4/20 |
| actor_critic-sp+biased/learned_static_ensemble | single_sub_policy | +2.6 | [-0.2, +6.2] | 12/20 |
| actor_critic-sp+biased/learned_static_ensemble | best_fixed_sub_policy | -15.9 | [-21.9, -8.9] | 1/20 |
| actor_critic-sp+biased/learned_static_ensemble | oracle_best_sub_policy | -30.0 | [-39.0, -21.6] | 0/20 |
| actor_critic-sp+biased/learned_static_ensemble | mappo | +68.1 | [+55.1, +79.3] | 20/20 |
| actor_critic-sp+biased/learned_static_ensemble | traj_s36 | +11.1 | [+5.2, +16.0] | 17/20 |
| actor_critic-sp+biased/learned_static_ensemble | hsp_s36 | +1.4 | [-11.3, +12.7] | 13/20 |
| actor_critic-sp+biased/learned_static_ensemble | fcp_s36 | +0.0 | [-4.4, +4.9] | 9/20 |
| actor_critic-sp+biased/learned_static_ensemble | mep_s36 | -2.9 | [-9.7, +5.2] | 6/20 |
| actor_critic-sp+biased/learned_static_ensemble | cole_s75 | +13.6 | [+4.6, +23.4] | 18/20 |
| actor_critic-sp+biased/learned_static_ensemble | e3t | +12.1 | [+1.9, +23.1] | 12/20 |
| actor_critic-sp+biased/learned_static_ensemble | mappo_argmax | +70.5 | [+57.2, +82.4] | 20/20 |
| actor_critic-sp+biased/learned_static_ensemble | traj_s36_argmax | +15.0 | [+7.3, +22.2] | 17/20 |
| actor_critic-sp+biased/learned_static_ensemble | hsp_s36_argmax | +3.9 | [-9.4, +15.9] | 13/20 |
| actor_critic-sp+biased/learned_static_ensemble | fcp_s36_argmax | +0.4 | [-5.8, +7.6] | 8/20 |
| actor_critic-sp+biased/learned_static_ensemble | mep_s36_argmax | +2.8 | [-4.1, +11.0] | 8/20 |
| actor_critic-sp+biased/learned_static_ensemble | cole_s75_argmax | +17.4 | [+8.7, +27.8] | 17/20 |
| actor_critic-sp+biased/learned_static_ensemble | e3t_argmax | +15.0 | [+3.6, +27.2] | 13/20 |
| actor_critic-sp+biased/learned_static_ensemble | best_baseline | -2.9 | [-9.7, +5.2] | 6/20 |
| actor_critic-sp+biased/learned_static_ensemble | best_baseline_argmax | +0.4 | [-5.8, +7.6] | 8/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | actor_critic-sp+biased-memoryless/learned_static_ensemble | +2.5 | [-1.5, +6.5] | 15/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | actor_critic-sp+biased-memoryless/top_learned_sub_policy | -6.2 | [-13.6, +1.2] | 7/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | static_ensemble | +9.5 | [+5.1, +14.0] | 17/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | static_ensemble_argmax | -12.7 | [-19.1, -5.8] | 3/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | single_sub_policy | +5.9 | [+0.7, +11.0] | 15/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | best_fixed_sub_policy | -12.6 | [-19.7, -4.9] | 3/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | oracle_best_sub_policy | -26.7 | [-34.5, -18.8] | 0/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | mappo | +71.4 | [+56.7, +85.6] | 20/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | traj_s36 | +14.4 | [+6.2, +22.0] | 16/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | hsp_s36 | +4.7 | [-6.3, +14.5] | 14/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | fcp_s36 | +3.3 | [-3.0, +9.4] | 11/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | mep_s36 | +0.4 | [-8.2, +9.9] | 8/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | cole_s75 | +16.9 | [+7.5, +27.3] | 18/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | e3t | +15.4 | [+7.3, +24.8] | 17/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | mappo_argmax | +73.9 | [+58.8, +88.6] | 20/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | traj_s36_argmax | +18.3 | [+8.3, +27.6] | 16/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | hsp_s36_argmax | +7.3 | [-4.6, +18.4] | 14/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | fcp_s36_argmax | +3.7 | [-4.0, +11.8] | 11/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | mep_s36_argmax | +6.1 | [-2.7, +15.8] | 10/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | cole_s75_argmax | +20.7 | [+11.2, +31.7] | 19/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | e3t_argmax | +18.3 | [+8.9, +29.3] | 18/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | best_baseline | +0.4 | [-8.2, +9.9] | 8/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | best_baseline_argmax | +3.7 | [-4.0, +11.8] | 11/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | actor_critic-sp+biased-memoryless/learned_static_ensemble | +9.6 | [+3.4, +15.3] | 15/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | actor_critic-sp+biased-memoryless/top_learned_sub_policy | +0.8 | [-6.8, +7.3] | 13/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | static_ensemble | +16.5 | [+10.4, +22.5] | 15/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | static_ensemble_argmax | -5.7 | [-8.9, -2.2] | 5/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | single_sub_policy | +13.0 | [+7.6, +18.1] | 16/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | best_fixed_sub_policy | -5.5 | [-14.1, +1.7] | 10/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | oracle_best_sub_policy | -19.6 | [-29.7, -9.7] | 4/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | mappo | +78.5 | [+62.7, +93.3] | 20/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | traj_s36 | +21.5 | [+14.5, +28.0] | 16/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | hsp_s36 | +11.8 | [-0.8, +23.5] | 13/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | fcp_s36 | +10.4 | [+4.8, +15.4] | 15/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | mep_s36 | +7.5 | [-1.0, +15.9] | 15/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | cole_s75 | +24.0 | [+14.7, +33.3] | 17/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | e3t | +22.5 | [+14.9, +30.0] | 19/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | mappo_argmax | +80.9 | [+64.8, +96.2] | 20/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | traj_s36_argmax | +25.4 | [+17.3, +32.8] | 18/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | hsp_s36_argmax | +14.3 | [+1.9, +25.8] | 14/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | fcp_s36_argmax | +10.7 | [+4.8, +16.3] | 17/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | mep_s36_argmax | +13.2 | [+4.6, +21.6] | 16/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | cole_s75_argmax | +27.8 | [+18.9, +37.2] | 18/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | e3t_argmax | +25.4 | [+17.1, +34.0] | 20/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | best_baseline | +7.5 | [-1.0, +15.9] | 15/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | best_baseline_argmax | +10.7 | [+4.8, +16.3] | 17/20 |
| actor_critic-sp+biased-memoryless/policy_selection | actor_critic-sp+biased-memoryless/learned_static_ensemble | +0.2 | [-8.0, +7.0] | 13/20 |
| actor_critic-sp+biased-memoryless/policy_selection | actor_critic-sp+biased-memoryless/top_learned_sub_policy | -8.6 | [-18.8, +0.0] | 7/20 |
| actor_critic-sp+biased-memoryless/policy_selection | static_ensemble | +7.1 | [-0.8, +14.1] | 16/20 |
| actor_critic-sp+biased-memoryless/policy_selection | static_ensemble_argmax | -15.1 | [-22.7, -7.9] | 2/20 |
| actor_critic-sp+biased-memoryless/policy_selection | single_sub_policy | +3.5 | [-4.7, +10.3] | 15/20 |
| actor_critic-sp+biased-memoryless/policy_selection | best_fixed_sub_policy | -15.0 | [-24.3, -6.4] | 2/20 |
| actor_critic-sp+biased-memoryless/policy_selection | oracle_best_sub_policy | -29.1 | [-38.3, -19.4] | 0/20 |
| actor_critic-sp+biased-memoryless/policy_selection | mappo | +69.0 | [+53.0, +84.7] | 20/20 |
| actor_critic-sp+biased-memoryless/policy_selection | traj_s36 | +12.0 | [+2.6, +20.7] | 16/20 |
| actor_critic-sp+biased-memoryless/policy_selection | hsp_s36 | +2.4 | [-9.6, +14.1] | 12/20 |
| actor_critic-sp+biased-memoryless/policy_selection | fcp_s36 | +1.0 | [-7.4, +8.4] | 12/20 |
| actor_critic-sp+biased-memoryless/policy_selection | mep_s36 | -2.0 | [-12.6, +8.3] | 9/20 |
| actor_critic-sp+biased-memoryless/policy_selection | cole_s75 | +14.5 | [+4.2, +25.0] | 16/20 |
| actor_critic-sp+biased-memoryless/policy_selection | e3t | +13.1 | [+6.2, +21.3] | 18/20 |
| actor_critic-sp+biased-memoryless/policy_selection | mappo_argmax | +71.5 | [+55.0, +87.8] | 20/20 |
| actor_critic-sp+biased-memoryless/policy_selection | traj_s36_argmax | +16.0 | [+4.8, +26.2] | 16/20 |
| actor_critic-sp+biased-memoryless/policy_selection | hsp_s36_argmax | +4.9 | [-8.0, +17.3] | 13/20 |
| actor_critic-sp+biased-memoryless/policy_selection | fcp_s36_argmax | +1.3 | [-8.2, +10.3] | 12/20 |
| actor_critic-sp+biased-memoryless/policy_selection | mep_s36_argmax | +3.7 | [-6.3, +13.7] | 11/20 |
| actor_critic-sp+biased-memoryless/policy_selection | cole_s75_argmax | +18.3 | [+7.8, +28.9] | 17/20 |
| actor_critic-sp+biased-memoryless/policy_selection | e3t_argmax | +15.9 | [+7.9, +25.8] | 18/20 |
| actor_critic-sp+biased-memoryless/policy_selection | best_baseline | -2.0 | [-12.6, +8.3] | 9/20 |
| actor_critic-sp+biased-memoryless/policy_selection | best_baseline_argmax | +1.3 | [-8.2, +10.3] | 12/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | static_ensemble | +6.9 | [+4.8, +8.6] | 19/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | static_ensemble_argmax | -15.3 | [-21.6, -8.2] | 4/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | single_sub_policy | +3.4 | [+0.5, +6.9] | 12/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | best_fixed_sub_policy | -15.1 | [-21.2, -8.2] | 1/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | oracle_best_sub_policy | -29.2 | [-38.1, -21.0] | 0/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | mappo | +68.9 | [+55.7, +80.4] | 20/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | traj_s36 | +11.9 | [+5.6, +16.9] | 17/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | hsp_s36 | +2.2 | [-10.4, +13.2] | 12/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | fcp_s36 | +0.8 | [-3.7, +5.5] | 11/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | mep_s36 | -2.1 | [-9.0, +6.0] | 8/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | cole_s75 | +14.4 | [+5.5, +24.1] | 18/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | e3t | +12.9 | [+2.8, +23.8] | 13/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | mappo_argmax | +71.3 | [+57.7, +83.4] | 20/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | traj_s36_argmax | +15.8 | [+7.7, +23.0] | 17/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | hsp_s36_argmax | +4.7 | [-8.4, +16.4] | 13/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | fcp_s36_argmax | +1.1 | [-4.9, +8.2] | 8/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | mep_s36_argmax | +3.6 | [-3.6, +11.9] | 9/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | cole_s75_argmax | +18.2 | [+9.3, +28.6] | 18/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | e3t_argmax | +15.8 | [+4.7, +27.8] | 14/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | best_baseline | -2.1 | [-9.0, +6.0] | 8/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | best_baseline_argmax | +1.1 | [-4.9, +8.2] | 8/20 |

## Ablations on biased partners (complete encoder - ablated, 95% bootstrap CI over distinct partner agents)

| ablated run | difference | 95% CI | partners better |
|---|---:|---|---:|
| actor_critic-sp+biased-memoryless/adaptive_ensemble | +5.8 | [+3.7, +8.1] | 19/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | +6.2 | [+2.2, +10.1] | 15/20 |
| actor_critic-sp+biased-memoryless/policy_selection | +9.0 | [+5.2, +12.9] | 19/20 |

Saved to /cmlscratch/kpowell1/adhoc_teamwork/runs/random3/s2-mix/analysis
