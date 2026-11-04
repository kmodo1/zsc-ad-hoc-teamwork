
## unident_s / hsp-w1 / partners: biased (20)

| method | mean return | std across seeds | seeds |
|---|---:|---:|---:|
| mep_s36_argmax | 308.6 | 1.8 | 5 |
| fcp_s36_argmax | 307.3 | 4.4 | 5 |
| mep_s36 | 305.6 | 2.3 | 5 |
| fcp_s36 | 304.7 | 3.3 | 5 |
| traj_s36_argmax | 301.7 | 7.6 | 5 |
| traj_s36 | 298.9 | 6.8 | 5 |
| hsp_s36_argmax | 293.7 | 26.8 | 5 |
| hsp_s36 | 290.5 | 17.0 | 5 |
| oracle_best_sub_policy | 278.7 | 0.0 | 1 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | 269.0 | 5.0 | 3 |
| actor_critic-sp+biased/adaptive_ensemble | 267.6 | 1.0 | 3 |
| actor_critic-sp+biased/policy_selection | 266.2 | 3.4 | 3 |
| static_ensemble_argmax | 236.0 | 0.0 | 1 |
| best_fixed_sub_policy | 217.7 | 0.0 | 1 |
| actor_critic-sp+biased/learned_static_ensemble | 195.1 | 3.8 | 3 |
| static_ensemble | 192.8 | 0.0 | 1 |
| actor_critic-sp+biased/top_learned_sub_policy | 188.5 | 12.6 | 3 |
| mappo | 179.8 | 17.4 | 5 |
| single_sub_policy | 171.3 | 35.9 | 10 |
| mappo_argmax | 165.5 | 11.5 | 5 |

## unident_s / hsp-w1 / partners: mep (10)

| method | mean return | std across seeds | seeds |
|---|---:|---:|---:|
| fcp_s36_argmax | 488.8 | 2.7 | 5 |
| mep_s36_argmax | 483.3 | 7.8 | 5 |
| fcp_s36 | 477.9 | 4.5 | 5 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | 476.4 | 31.0 | 3 |
| mep_s36 | 475.9 | 6.2 | 5 |
| hsp_s36_argmax | 473.2 | 28.2 | 5 |
| hsp_s36 | 471.2 | 8.4 | 5 |
| actor_critic-sp+biased/adaptive_ensemble | 467.0 | 7.2 | 3 |
| actor_critic-sp+biased/policy_selection | 449.1 | 25.6 | 3 |
| oracle_best_sub_policy | 445.8 | 0.0 | 1 |
| mappo | 440.2 | 69.4 | 5 |
| best_fixed_sub_policy | 439.6 | 0.0 | 1 |
| static_ensemble_argmax | 439.6 | 0.0 | 1 |
| traj_s36_argmax | 420.3 | 22.6 | 5 |
| traj_s36 | 417.4 | 20.4 | 5 |
| mappo_argmax | 384.2 | 54.2 | 5 |
| actor_critic-sp+biased/learned_static_ensemble | 366.1 | 4.6 | 3 |
| static_ensemble | 356.1 | 0.0 | 1 |
| single_sub_policy | 334.3 | 81.2 | 10 |
| actor_critic-sp+biased/top_learned_sub_policy | 284.7 | 104.6 | 3 |

## unident_s / hsp-w1 / partners: self_play (5)

| method | mean return | std across seeds | seeds |
|---|---:|---:|---:|
| fcp_s36_argmax | 478.5 | 4.7 | 5 |
| hsp_s36_argmax | 477.9 | 15.2 | 5 |
| fcp_s36 | 474.4 | 2.9 | 5 |
| mep_s36_argmax | 473.0 | 9.3 | 5 |
| hsp_s36 | 467.6 | 11.7 | 5 |
| mep_s36 | 465.5 | 9.5 | 5 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | 449.7 | 11.4 | 3 |
| traj_s36_argmax | 448.3 | 10.0 | 5 |
| actor_critic-sp+biased/adaptive_ensemble | 448.2 | 3.7 | 3 |
| oracle_best_sub_policy | 444.4 | 0.0 | 1 |
| actor_critic-sp+biased/policy_selection | 441.4 | 11.3 | 3 |
| best_fixed_sub_policy | 440.8 | 0.0 | 1 |
| mappo | 440.5 | 41.7 | 5 |
| traj_s36 | 437.8 | 11.3 | 5 |
| static_ensemble_argmax | 431.2 | 0.0 | 1 |
| mappo_argmax | 430.6 | 46.6 | 5 |
| actor_critic-sp+biased/learned_static_ensemble | 371.5 | 5.0 | 3 |
| static_ensemble | 363.3 | 0.0 | 1 |
| single_sub_policy | 348.4 | 70.4 | 10 |
| actor_critic-sp+biased/top_learned_sub_policy | 335.3 | 70.1 | 3 |

## Differences on biased partners (ours - reference, 95% bootstrap CI over distinct partner agents)

| ours | reference | difference | 95% CI | partners better |
|---|---|---:|---|---:|
| actor_critic-sp+biased/adaptive_ensemble | actor_critic-sp+biased/learned_static_ensemble | +72.5 | [+56.4, +88.8] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble | actor_critic-sp+biased/top_learned_sub_policy | +79.1 | [+48.3, +113.2] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble | static_ensemble | +74.8 | [+57.6, +94.9] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble | static_ensemble_argmax | +31.6 | [+12.7, +50.5] | 15/20 |
| actor_critic-sp+biased/adaptive_ensemble | single_sub_policy | +96.4 | [+79.4, +111.3] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble | best_fixed_sub_policy | +49.9 | [+26.2, +72.1] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble | oracle_best_sub_policy | -11.1 | [-26.4, +4.4] | 7/20 |
| actor_critic-sp+biased/adaptive_ensemble | mappo | +87.8 | [+48.5, +126.6] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble | traj_s36 | -31.2 | [-56.3, -9.0] | 4/20 |
| actor_critic-sp+biased/adaptive_ensemble | hsp_s36 | -22.9 | [-35.1, -9.9] | 4/20 |
| actor_critic-sp+biased/adaptive_ensemble | fcp_s36 | -37.1 | [-56.8, -19.3] | 2/20 |
| actor_critic-sp+biased/adaptive_ensemble | mep_s36 | -37.9 | [-59.6, -19.8] | 1/20 |
| actor_critic-sp+biased/adaptive_ensemble | mappo_argmax | +102.1 | [+59.9, +144.3] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble | traj_s36_argmax | -34.1 | [-59.3, -11.4] | 4/20 |
| actor_critic-sp+biased/adaptive_ensemble | hsp_s36_argmax | -26.0 | [-39.9, -11.5] | 2/20 |
| actor_critic-sp+biased/adaptive_ensemble | fcp_s36_argmax | -39.6 | [-60.1, -20.6] | 2/20 |
| actor_critic-sp+biased/adaptive_ensemble | mep_s36_argmax | -40.9 | [-63.3, -21.6] | 2/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | actor_critic-sp+biased/learned_static_ensemble | +73.9 | [+50.1, +94.1] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | actor_critic-sp+biased/top_learned_sub_policy | +80.5 | [+51.7, +113.9] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | static_ensemble | +76.2 | [+51.5, +99.3] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | static_ensemble_argmax | +33.0 | [+6.5, +52.9] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | single_sub_policy | +97.7 | [+76.7, +117.1] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | best_fixed_sub_policy | +51.2 | [+18.2, +76.6] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | oracle_best_sub_policy | -9.8 | [-30.9, +10.4] | 8/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mappo | +89.2 | [+51.4, +129.3] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | traj_s36 | -29.9 | [-57.5, -4.9] | 4/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | hsp_s36 | -21.5 | [-36.4, -6.5] | 8/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | fcp_s36 | -35.8 | [-57.6, -16.9] | 4/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mep_s36 | -36.6 | [-61.5, -15.5] | 5/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mappo_argmax | +103.4 | [+63.0, +146.2] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | traj_s36_argmax | -32.7 | [-60.2, -8.9] | 4/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | hsp_s36_argmax | -24.7 | [-38.7, -8.5] | 7/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | fcp_s36_argmax | -38.3 | [-60.2, -19.7] | 3/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mep_s36_argmax | -39.6 | [-64.5, -18.1] | 4/20 |
| actor_critic-sp+biased/policy_selection | actor_critic-sp+biased/learned_static_ensemble | +71.1 | [+51.2, +90.5] | 19/20 |
| actor_critic-sp+biased/policy_selection | actor_critic-sp+biased/top_learned_sub_policy | +77.7 | [+51.1, +106.9] | 20/20 |
| actor_critic-sp+biased/policy_selection | static_ensemble | +73.4 | [+51.2, +96.7] | 19/20 |
| actor_critic-sp+biased/policy_selection | static_ensemble_argmax | +30.2 | [+8.2, +49.6] | 17/20 |
| actor_critic-sp+biased/policy_selection | single_sub_policy | +94.9 | [+75.4, +113.2] | 20/20 |
| actor_critic-sp+biased/policy_selection | best_fixed_sub_policy | +48.5 | [+18.8, +75.1] | 19/20 |
| actor_critic-sp+biased/policy_selection | oracle_best_sub_policy | -12.5 | [-31.4, +6.4] | 10/20 |
| actor_critic-sp+biased/policy_selection | mappo | +86.4 | [+48.8, +125.1] | 19/20 |
| actor_critic-sp+biased/policy_selection | traj_s36 | -32.7 | [-58.7, -10.1] | 5/20 |
| actor_critic-sp+biased/policy_selection | hsp_s36 | -24.3 | [-37.7, -10.1] | 4/20 |
| actor_critic-sp+biased/policy_selection | fcp_s36 | -38.6 | [-59.2, -21.1] | 3/20 |
| actor_critic-sp+biased/policy_selection | mep_s36 | -39.4 | [-62.5, -20.5] | 3/20 |
| actor_critic-sp+biased/policy_selection | mappo_argmax | +100.7 | [+60.7, +141.5] | 19/20 |
| actor_critic-sp+biased/policy_selection | traj_s36_argmax | -35.5 | [-61.6, -13.2] | 5/20 |
| actor_critic-sp+biased/policy_selection | hsp_s36_argmax | -27.5 | [-40.4, -12.9] | 4/20 |
| actor_critic-sp+biased/policy_selection | fcp_s36_argmax | -41.1 | [-62.1, -22.8] | 2/20 |
| actor_critic-sp+biased/policy_selection | mep_s36_argmax | -42.4 | [-66.3, -22.6] | 3/20 |
| actor_critic-sp+biased/learned_static_ensemble | static_ensemble | +2.3 | [-10.2, +13.9] | 11/20 |
| actor_critic-sp+biased/learned_static_ensemble | static_ensemble_argmax | -40.9 | [-54.5, -23.9] | 2/20 |
| actor_critic-sp+biased/learned_static_ensemble | single_sub_policy | +23.8 | [+13.8, +34.5] | 19/20 |
| actor_critic-sp+biased/learned_static_ensemble | best_fixed_sub_policy | -22.6 | [-36.5, -10.8] | 4/20 |
| actor_critic-sp+biased/learned_static_ensemble | oracle_best_sub_policy | -83.6 | [-102.1, -65.6] | 0/20 |
| actor_critic-sp+biased/learned_static_ensemble | mappo | +15.3 | [-18.7, +45.7] | 13/20 |
| actor_critic-sp+biased/learned_static_ensemble | traj_s36 | -103.8 | [-132.7, -74.4] | 0/20 |
| actor_critic-sp+biased/learned_static_ensemble | hsp_s36 | -95.4 | [-115.1, -71.5] | 0/20 |
| actor_critic-sp+biased/learned_static_ensemble | fcp_s36 | -109.6 | [-133.4, -83.7] | 0/20 |
| actor_critic-sp+biased/learned_static_ensemble | mep_s36 | -110.5 | [-136.3, -82.9] | 0/20 |
| actor_critic-sp+biased/learned_static_ensemble | mappo_argmax | +29.6 | [-8.3, +63.3] | 14/20 |
| actor_critic-sp+biased/learned_static_ensemble | traj_s36_argmax | -106.6 | [-135.7, -77.0] | 0/20 |
| actor_critic-sp+biased/learned_static_ensemble | hsp_s36_argmax | -98.6 | [-120.5, -73.0] | 0/20 |
| actor_critic-sp+biased/learned_static_ensemble | fcp_s36_argmax | -112.2 | [-136.4, -84.8] | 0/20 |
| actor_critic-sp+biased/learned_static_ensemble | mep_s36_argmax | -113.5 | [-139.2, -86.0] | 0/20 |

Saved to /cmlscratch/kpowell1/adhoc_teamwork/runs/unident_s/hsp-w1/analysis
