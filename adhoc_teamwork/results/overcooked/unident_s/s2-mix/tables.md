
## unident_s / s2-mix / partners: biased (20)

| method | mean return | std across seeds | seeds |
|---|---:|---:|---:|
| oracle_best_sub_policy | 323.0 | 0.0 | 1 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | 321.0 | 2.3 | 3 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | 317.7 | 1.0 | 3 |
| static_ensemble_argmax | 316.9 | 0.0 | 1 |
| actor_critic-sp+biased/policy_selection | 311.5 | 0.9 | 3 |
| best_fixed_sub_policy | 310.2 | 0.0 | 1 |
| actor_critic-sp+biased-memoryless/top_learned_sub_policy | 308.9 | 1.1 | 3 |
| mep_s36_argmax | 308.6 | 1.8 | 5 |
| actor_critic-sp+biased/top_learned_sub_policy | 308.2 | 0.0 | 3 |
| fcp_s36_argmax | 307.3 | 4.4 | 5 |
| mep_s36 | 305.6 | 2.3 | 5 |
| fcp_s36 | 304.7 | 3.3 | 5 |
| actor_critic-sp+biased-memoryless/policy_selection | 302.0 | 1.0 | 3 |
| single_sub_policy | 301.7 | 8.8 | 12 |
| traj_s36_argmax | 301.7 | 7.6 | 5 |
| actor_critic-sp+biased/adaptive_ensemble | 301.2 | 1.1 | 3 |
| traj_s36 | 298.9 | 6.8 | 5 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | 298.4 | 1.1 | 3 |
| actor_critic-sp+biased/learned_static_ensemble | 297.5 | 1.1 | 3 |
| static_ensemble | 297.1 | 0.0 | 1 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | 296.6 | 0.6 | 3 |
| hsp_s36_argmax | 293.7 | 26.8 | 5 |
| hsp_s36 | 290.5 | 17.0 | 5 |
| mappo | 179.8 | 17.4 | 5 |
| mappo_argmax | 165.5 | 11.5 | 5 |

## Differences on biased partners (ours - reference, 95% bootstrap CI over distinct partner agents)

| ours | reference | difference | 95% CI | partners better |
|---|---|---:|---|---:|
| actor_critic-sp+biased/adaptive_ensemble | actor_critic-sp+biased/learned_static_ensemble | +3.7 | [+1.8, +5.3] | 17/20 |
| actor_critic-sp+biased/adaptive_ensemble | actor_critic-sp+biased/top_learned_sub_policy | -7.0 | [-17.3, +4.2] | 8/20 |
| actor_critic-sp+biased/adaptive_ensemble | static_ensemble | +4.1 | [+2.2, +5.9] | 16/20 |
| actor_critic-sp+biased/adaptive_ensemble | static_ensemble_argmax | -15.7 | [-26.6, -2.7] | 2/20 |
| actor_critic-sp+biased/adaptive_ensemble | single_sub_policy | -0.5 | [-3.3, +3.4] | 8/20 |
| actor_critic-sp+biased/adaptive_ensemble | best_fixed_sub_policy | -9.0 | [-13.7, -4.9] | 3/20 |
| actor_critic-sp+biased/adaptive_ensemble | oracle_best_sub_policy | -21.8 | [-26.7, -17.1] | 0/20 |
| actor_critic-sp+biased/adaptive_ensemble | mappo | +121.4 | [+74.5, +169.2] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble | traj_s36 | +2.3 | [-3.8, +9.6] | 10/20 |
| actor_critic-sp+biased/adaptive_ensemble | hsp_s36 | +10.7 | [-1.0, +24.2] | 14/20 |
| actor_critic-sp+biased/adaptive_ensemble | fcp_s36 | -3.5 | [-7.3, +0.8] | 4/20 |
| actor_critic-sp+biased/adaptive_ensemble | mep_s36 | -4.4 | [-5.9, -3.0] | 2/20 |
| actor_critic-sp+biased/adaptive_ensemble | mappo_argmax | +135.7 | [+86.9, +185.9] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble | traj_s36_argmax | -0.5 | [-6.5, +6.4] | 7/20 |
| actor_critic-sp+biased/adaptive_ensemble | hsp_s36_argmax | +7.5 | [-6.3, +25.1] | 13/20 |
| actor_critic-sp+biased/adaptive_ensemble | fcp_s36_argmax | -6.1 | [-11.0, -0.8] | 3/20 |
| actor_critic-sp+biased/adaptive_ensemble | mep_s36_argmax | -7.4 | [-10.0, -4.6] | 1/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | actor_critic-sp+biased/learned_static_ensemble | +23.5 | [+16.8, +29.8] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | actor_critic-sp+biased/top_learned_sub_policy | +12.8 | [+2.7, +21.9] | 14/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | static_ensemble | +23.9 | [+17.8, +30.2] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | static_ensemble_argmax | +4.1 | [-2.6, +12.1] | 10/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | single_sub_policy | +19.3 | [+15.2, +24.7] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | best_fixed_sub_policy | +10.8 | [+3.6, +17.2] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | oracle_best_sub_policy | -2.0 | [-9.2, +4.8] | 9/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mappo | +141.2 | [+98.0, +184.4] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | traj_s36 | +22.1 | [+11.8, +32.5] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | hsp_s36 | +30.5 | [+20.7, +40.1] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | fcp_s36 | +16.3 | [+11.1, +22.3] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mep_s36 | +15.4 | [+8.3, +23.4] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mappo_argmax | +155.5 | [+111.2, +201.0] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | traj_s36_argmax | +19.3 | [+8.8, +30.0] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | hsp_s36_argmax | +27.3 | [+16.1, +40.3] | 17/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | fcp_s36_argmax | +13.7 | [+7.2, +21.1] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mep_s36_argmax | +12.4 | [+5.1, +20.7] | 17/20 |
| actor_critic-sp+biased/policy_selection | actor_critic-sp+biased/learned_static_ensemble | +14.0 | [+8.4, +21.0] | 19/20 |
| actor_critic-sp+biased/policy_selection | actor_critic-sp+biased/top_learned_sub_policy | +3.2 | [-3.7, +11.1] | 10/20 |
| actor_critic-sp+biased/policy_selection | static_ensemble | +14.4 | [+9.5, +21.3] | 19/20 |
| actor_critic-sp+biased/policy_selection | static_ensemble_argmax | -5.4 | [-17.0, +6.8] | 5/20 |
| actor_critic-sp+biased/policy_selection | single_sub_policy | +9.7 | [+5.2, +16.2] | 18/20 |
| actor_critic-sp+biased/policy_selection | best_fixed_sub_policy | +1.3 | [-6.0, +8.3] | 9/20 |
| actor_critic-sp+biased/policy_selection | oracle_best_sub_policy | -11.5 | [-15.1, -8.1] | 0/20 |
| actor_critic-sp+biased/policy_selection | mappo | +131.7 | [+88.1, +176.4] | 20/20 |
| actor_critic-sp+biased/policy_selection | traj_s36 | +12.6 | [+2.4, +25.2] | 11/20 |
| actor_critic-sp+biased/policy_selection | hsp_s36 | +21.0 | [+12.0, +31.4] | 19/20 |
| actor_critic-sp+biased/policy_selection | fcp_s36 | +6.7 | [+1.1, +13.3] | 12/20 |
| actor_critic-sp+biased/policy_selection | mep_s36 | +5.9 | [+1.6, +12.2] | 15/20 |
| actor_critic-sp+biased/policy_selection | mappo_argmax | +146.0 | [+99.7, +193.1] | 20/20 |
| actor_critic-sp+biased/policy_selection | traj_s36_argmax | +9.8 | [-0.3, +21.3] | 10/20 |
| actor_critic-sp+biased/policy_selection | hsp_s36_argmax | +17.8 | [+5.5, +32.9] | 17/20 |
| actor_critic-sp+biased/policy_selection | fcp_s36_argmax | +4.2 | [-1.6, +10.5] | 10/20 |
| actor_critic-sp+biased/policy_selection | mep_s36_argmax | +2.9 | [-1.9, +9.5] | 13/20 |
| actor_critic-sp+biased/learned_static_ensemble | static_ensemble | +0.4 | [-1.0, +1.6] | 11/20 |
| actor_critic-sp+biased/learned_static_ensemble | static_ensemble_argmax | -19.4 | [-29.7, -6.7] | 1/20 |
| actor_critic-sp+biased/learned_static_ensemble | single_sub_policy | -4.2 | [-7.3, -0.2] | 2/20 |
| actor_critic-sp+biased/learned_static_ensemble | best_fixed_sub_policy | -12.7 | [-16.1, -9.6] | 1/20 |
| actor_critic-sp+biased/learned_static_ensemble | oracle_best_sub_policy | -25.5 | [-30.9, -20.7] | 0/20 |
| actor_critic-sp+biased/learned_static_ensemble | mappo | +117.7 | [+71.2, +165.3] | 18/20 |
| actor_critic-sp+biased/learned_static_ensemble | traj_s36 | -1.4 | [-7.6, +5.5] | 7/20 |
| actor_critic-sp+biased/learned_static_ensemble | hsp_s36 | +7.0 | [-5.0, +20.5] | 12/20 |
| actor_critic-sp+biased/learned_static_ensemble | fcp_s36 | -7.3 | [-10.4, -3.3] | 2/20 |
| actor_critic-sp+biased/learned_static_ensemble | mep_s36 | -8.1 | [-10.4, -5.3] | 2/20 |
| actor_critic-sp+biased/learned_static_ensemble | mappo_argmax | +132.0 | [+83.6, +182.0] | 19/20 |
| actor_critic-sp+biased/learned_static_ensemble | traj_s36_argmax | -4.2 | [-10.1, +2.2] | 7/20 |
| actor_critic-sp+biased/learned_static_ensemble | hsp_s36_argmax | +3.8 | [-10.1, +21.5] | 13/20 |
| actor_critic-sp+biased/learned_static_ensemble | fcp_s36_argmax | -9.8 | [-14.2, -4.9] | 2/20 |
| actor_critic-sp+biased/learned_static_ensemble | mep_s36_argmax | -11.1 | [-13.8, -8.0] | 3/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | actor_critic-sp+biased-memoryless/learned_static_ensemble | +1.8 | [+0.6, +3.0] | 16/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | actor_critic-sp+biased-memoryless/top_learned_sub_policy | -10.5 | [-12.6, -8.5] | 0/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | static_ensemble | +1.3 | [-0.0, +2.6] | 14/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | static_ensemble_argmax | -18.5 | [-29.2, -5.9] | 1/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | single_sub_policy | -3.3 | [-6.3, +0.4] | 4/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | best_fixed_sub_policy | -11.8 | [-15.6, -8.2] | 1/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | oracle_best_sub_policy | -24.6 | [-30.1, -19.5] | 0/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | mappo | +118.6 | [+71.8, +166.5] | 18/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | traj_s36 | -0.5 | [-5.9, +5.9] | 6/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | hsp_s36 | +7.9 | [-4.3, +21.4] | 13/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | fcp_s36 | -6.3 | [-9.3, -2.8] | 2/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | mep_s36 | -7.1 | [-9.2, -4.8] | 1/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | mappo_argmax | +132.9 | [+84.2, +183.2] | 19/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | traj_s36_argmax | -3.3 | [-8.7, +2.7] | 6/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | hsp_s36_argmax | +4.7 | [-9.7, +22.2] | 13/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | fcp_s36_argmax | -8.9 | [-13.3, -4.2] | 2/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble | mep_s36_argmax | -10.2 | [-12.9, -7.2] | 1/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | actor_critic-sp+biased-memoryless/learned_static_ensemble | +21.1 | [+10.2, +30.6] | 19/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | actor_critic-sp+biased-memoryless/top_learned_sub_policy | +8.8 | [-1.6, +17.9] | 17/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | static_ensemble | +20.7 | [+9.2, +30.4] | 18/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | static_ensemble_argmax | +0.9 | [-1.2, +2.8] | 11/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | single_sub_policy | +16.0 | [+7.3, +24.1] | 18/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | best_fixed_sub_policy | +7.6 | [-4.8, +17.1] | 19/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | oracle_best_sub_policy | -5.2 | [-16.8, +4.4] | 12/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | mappo | +138.0 | [+96.0, +180.2] | 20/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | traj_s36 | +18.9 | [+4.0, +33.2] | 19/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | hsp_s36 | +27.3 | [+19.7, +34.7] | 18/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | fcp_s36 | +13.0 | [+3.8, +22.2] | 18/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | mep_s36 | +12.2 | [-0.3, +23.0] | 18/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | mappo_argmax | +152.2 | [+108.7, +196.9] | 20/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | traj_s36_argmax | +16.1 | [+1.3, +30.5] | 19/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | hsp_s36_argmax | +24.1 | [+16.2, +32.8] | 17/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | fcp_s36_argmax | +10.5 | [+1.4, +19.8] | 18/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | mep_s36_argmax | +9.2 | [-3.4, +20.6] | 17/20 |
| actor_critic-sp+biased-memoryless/policy_selection | actor_critic-sp+biased-memoryless/learned_static_ensemble | +5.4 | [+1.6, +8.7] | 17/20 |
| actor_critic-sp+biased-memoryless/policy_selection | actor_critic-sp+biased-memoryless/top_learned_sub_policy | -6.9 | [-9.9, -4.2] | 4/20 |
| actor_critic-sp+biased-memoryless/policy_selection | static_ensemble | +5.0 | [+0.7, +8.5] | 17/20 |
| actor_critic-sp+biased-memoryless/policy_selection | static_ensemble_argmax | -14.8 | [-24.1, -5.3] | 2/20 |
| actor_critic-sp+biased-memoryless/policy_selection | single_sub_policy | +0.3 | [-2.3, +3.0] | 12/20 |
| actor_critic-sp+biased-memoryless/policy_selection | best_fixed_sub_policy | -8.1 | [-13.9, -3.3] | 5/20 |
| actor_critic-sp+biased-memoryless/policy_selection | oracle_best_sub_policy | -20.9 | [-27.2, -14.6] | 0/20 |
| actor_critic-sp+biased-memoryless/policy_selection | mappo | +122.3 | [+77.3, +168.0] | 20/20 |
| actor_critic-sp+biased-memoryless/policy_selection | traj_s36 | +3.2 | [-4.1, +10.7] | 10/20 |
| actor_critic-sp+biased-memoryless/policy_selection | hsp_s36 | +11.6 | [+0.5, +22.6] | 14/20 |
| actor_critic-sp+biased-memoryless/policy_selection | fcp_s36 | -2.7 | [-5.2, -0.2] | 9/20 |
| actor_critic-sp+biased-memoryless/policy_selection | mep_s36 | -3.5 | [-8.6, +1.1] | 8/20 |
| actor_critic-sp+biased-memoryless/policy_selection | mappo_argmax | +136.5 | [+89.5, +184.6] | 20/20 |
| actor_critic-sp+biased-memoryless/policy_selection | traj_s36_argmax | +0.4 | [-7.0, +7.7] | 7/20 |
| actor_critic-sp+biased-memoryless/policy_selection | hsp_s36_argmax | +8.4 | [-4.5, +22.6] | 13/20 |
| actor_critic-sp+biased-memoryless/policy_selection | fcp_s36_argmax | -5.2 | [-8.9, -1.4] | 5/20 |
| actor_critic-sp+biased-memoryless/policy_selection | mep_s36_argmax | -6.5 | [-11.8, -1.5] | 5/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | static_ensemble | -0.5 | [-1.8, +0.9] | 8/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | static_ensemble_argmax | -20.3 | [-30.6, -8.5] | 1/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | single_sub_policy | -5.1 | [-7.3, -2.4] | 3/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | best_fixed_sub_policy | -13.6 | [-18.2, -9.4] | 1/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | oracle_best_sub_policy | -26.4 | [-31.9, -21.2] | 0/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | mappo | +116.8 | [+70.4, +164.6] | 18/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | traj_s36 | -2.3 | [-8.2, +4.5] | 5/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | hsp_s36 | +6.1 | [-5.4, +18.7] | 13/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | fcp_s36 | -8.1 | [-10.9, -5.1] | 2/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | mep_s36 | -8.9 | [-11.0, -6.9] | 0/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | mappo_argmax | +131.1 | [+82.5, +181.1] | 19/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | traj_s36_argmax | -5.1 | [-10.9, +1.5] | 6/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | hsp_s36_argmax | +3.0 | [-10.5, +19.4] | 13/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | fcp_s36_argmax | -10.6 | [-14.7, -6.6] | 2/20 |
| actor_critic-sp+biased-memoryless/learned_static_ensemble | mep_s36_argmax | -12.0 | [-14.4, -9.3] | 1/20 |

## Ablations on biased partners (complete encoder - ablated, 95% bootstrap CI over distinct partner agents)

| ablated run | difference | 95% CI | partners better |
|---|---:|---|---:|
| actor_critic-sp+biased-memoryless/adaptive_ensemble | +2.8 | [+1.4, +4.1] | 18/20 |
| actor_critic-sp+biased-memoryless/adaptive_ensemble_argmax | +3.3 | [-2.5, +10.0] | 8/20 |
| actor_critic-sp+biased-memoryless/policy_selection | +9.4 | [+3.3, +16.3] | 12/20 |

Saved to /cmlscratch/kpowell1/adhoc_teamwork/runs/unident_s/s2-mix/analysis
