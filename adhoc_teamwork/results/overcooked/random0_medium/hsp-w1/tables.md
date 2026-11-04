
## random0_medium / hsp-w1 / partners: biased (20)

| method | mean return | std across seeds | seeds |
|---|---:|---:|---:|
| oracle_best_sub_policy | 154.6 | 0.0 | 1 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | 152.7 | 3.4 | 3 |
| actor_critic-sp+biased/policy_selection | 145.7 | 3.0 | 3 |
| actor_critic-sp+biased/adaptive_ensemble | 142.8 | 3.0 | 3 |
| hsp_s36_argmax | 140.8 | 14.9 | 5 |
| hsp_s36 | 134.3 | 11.1 | 5 |
| mep_s36_argmax | 119.1 | 10.2 | 5 |
| mappo | 118.4 | 14.1 | 5 |
| fcp_s36 | 118.3 | 11.0 | 5 |
| mep_s36 | 116.3 | 12.2 | 5 |
| fcp_s36_argmax | 115.6 | 13.0 | 5 |
| best_fixed_sub_policy | 115.4 | 0.0 | 1 |
| static_ensemble_argmax | 115.2 | 0.0 | 1 |
| mappo_argmax | 113.5 | 17.9 | 5 |
| actor_critic-sp+biased/learned_static_ensemble | 109.6 | 1.5 | 3 |
| traj_s36_argmax | 104.0 | 19.9 | 5 |
| traj_s36 | 103.8 | 21.2 | 5 |
| actor_critic-sp+biased/top_learned_sub_policy | 101.7 | 0.0 | 3 |
| static_ensemble | 97.2 | 0.0 | 1 |
| single_sub_policy | 74.2 | 32.8 | 10 |

## random0_medium / hsp-w1 / partners: mep (10)

| method | mean return | std across seeds | seeds |
|---|---:|---:|---:|
| mappo | 326.9 | 5.0 | 5 |
| mappo_argmax | 323.4 | 5.9 | 5 |
| mep_s36_argmax | 320.6 | 15.1 | 5 |
| fcp_s36_argmax | 317.4 | 21.0 | 5 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | 316.7 | 3.0 | 3 |
| hsp_s36_argmax | 313.1 | 10.1 | 5 |
| fcp_s36 | 310.0 | 16.1 | 5 |
| mep_s36 | 309.9 | 22.8 | 5 |
| actor_critic-sp+biased/policy_selection | 304.0 | 4.8 | 3 |
| actor_critic-sp+biased/adaptive_ensemble | 302.3 | 4.0 | 3 |
| oracle_best_sub_policy | 298.8 | 0.0 | 1 |
| actor_critic-sp+biased/top_learned_sub_policy | 298.8 | 0.0 | 3 |
| hsp_s36 | 297.0 | 19.6 | 5 |
| traj_s36_argmax | 244.4 | 82.7 | 5 |
| traj_s36 | 240.4 | 79.4 | 5 |
| best_fixed_sub_policy | 203.1 | 0.0 | 1 |
| static_ensemble_argmax | 182.8 | 0.0 | 1 |
| actor_critic-sp+biased/learned_static_ensemble | 182.5 | 11.2 | 3 |
| static_ensemble | 132.6 | 0.0 | 1 |
| single_sub_policy | 103.8 | 99.2 | 10 |

## random0_medium / hsp-w1 / partners: self_play (5)

| method | mean return | std across seeds | seeds |
|---|---:|---:|---:|
| mappo_argmax | 338.7 | 8.5 | 5 |
| mappo | 337.2 | 5.9 | 5 |
| fcp_s36_argmax | 322.3 | 19.6 | 5 |
| mep_s36_argmax | 322.1 | 16.6 | 5 |
| fcp_s36 | 313.4 | 15.5 | 5 |
| hsp_s36_argmax | 311.4 | 15.2 | 5 |
| mep_s36 | 309.9 | 25.3 | 5 |
| actor_critic-sp+biased/policy_selection | 306.2 | 0.6 | 3 |
| oracle_best_sub_policy | 305.5 | 0.0 | 1 |
| actor_critic-sp+biased/top_learned_sub_policy | 305.5 | 0.0 | 3 |
| actor_critic-sp+biased/adaptive_ensemble | 305.2 | 0.8 | 3 |
| hsp_s36 | 298.6 | 18.8 | 5 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | 291.6 | 3.9 | 3 |
| traj_s36_argmax | 251.0 | 86.0 | 5 |
| traj_s36 | 240.0 | 83.1 | 5 |
| best_fixed_sub_policy | 194.2 | 0.0 | 1 |
| actor_critic-sp+biased/learned_static_ensemble | 175.4 | 16.4 | 3 |
| static_ensemble_argmax | 152.8 | 0.0 | 1 |
| static_ensemble | 122.7 | 0.0 | 1 |
| single_sub_policy | 103.5 | 104.7 | 10 |

## Differences on biased partners (ours - reference, 95% bootstrap CI over distinct partner agents)

| ours | reference | difference | 95% CI | partners better |
|---|---|---:|---|---:|
| actor_critic-sp+biased/adaptive_ensemble | actor_critic-sp+biased/learned_static_ensemble | +33.2 | [+20.2, +48.9] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble | actor_critic-sp+biased/top_learned_sub_policy | +41.1 | [+21.5, +63.5] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble | static_ensemble | +45.6 | [+21.3, +74.5] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble | static_ensemble_argmax | +27.6 | [+2.0, +56.9] | 13/20 |
| actor_critic-sp+biased/adaptive_ensemble | single_sub_policy | +68.6 | [+40.3, +101.3] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble | best_fixed_sub_policy | +27.4 | [+18.1, +39.8] | 17/20 |
| actor_critic-sp+biased/adaptive_ensemble | oracle_best_sub_policy | -11.8 | [-24.5, -0.2] | 6/20 |
| actor_critic-sp+biased/adaptive_ensemble | mappo | +24.4 | [+17.5, +30.7] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble | traj_s36 | +39.0 | [+29.4, +49.3] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble | hsp_s36 | +8.5 | [-4.5, +20.3] | 14/20 |
| actor_critic-sp+biased/adaptive_ensemble | fcp_s36 | +24.5 | [+16.4, +34.0] | 17/20 |
| actor_critic-sp+biased/adaptive_ensemble | mep_s36 | +26.5 | [+15.3, +37.9] | 17/20 |
| actor_critic-sp+biased/adaptive_ensemble | mappo_argmax | +29.3 | [+21.6, +36.2] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble | traj_s36_argmax | +38.8 | [+30.3, +48.4] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble | hsp_s36_argmax | +2.0 | [-9.1, +12.1] | 10/20 |
| actor_critic-sp+biased/adaptive_ensemble | fcp_s36_argmax | +27.2 | [+18.4, +37.3] | 17/20 |
| actor_critic-sp+biased/adaptive_ensemble | mep_s36_argmax | +23.7 | [+10.8, +37.1] | 16/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | actor_critic-sp+biased/learned_static_ensemble | +43.1 | [+28.0, +61.6] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | actor_critic-sp+biased/top_learned_sub_policy | +51.1 | [+30.0, +75.3] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | static_ensemble | +55.6 | [+29.7, +86.9] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | static_ensemble_argmax | +37.5 | [+11.7, +68.6] | 14/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | single_sub_policy | +78.5 | [+48.3, +113.8] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | best_fixed_sub_policy | +37.3 | [+24.4, +52.4] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | oracle_best_sub_policy | -1.9 | [-15.8, +11.3] | 10/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mappo | +34.4 | [+23.2, +43.8] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | traj_s36 | +48.9 | [+37.0, +60.8] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | hsp_s36 | +18.4 | [+2.3, +33.5] | 14/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | fcp_s36 | +34.4 | [+23.1, +46.3] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mep_s36 | +36.4 | [+23.0, +50.1] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mappo_argmax | +39.2 | [+26.8, +49.9] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | traj_s36_argmax | +48.7 | [+38.6, +59.4] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | hsp_s36_argmax | +12.0 | [-2.7, +25.4] | 14/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | fcp_s36_argmax | +37.1 | [+25.3, +49.3] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mep_s36_argmax | +33.7 | [+19.1, +49.1] | 17/20 |
| actor_critic-sp+biased/policy_selection | actor_critic-sp+biased/learned_static_ensemble | +36.1 | [+21.1, +54.0] | 19/20 |
| actor_critic-sp+biased/policy_selection | actor_critic-sp+biased/top_learned_sub_policy | +44.1 | [+26.2, +65.6] | 20/20 |
| actor_critic-sp+biased/policy_selection | static_ensemble | +48.6 | [+22.5, +79.6] | 20/20 |
| actor_critic-sp+biased/policy_selection | static_ensemble_argmax | +30.5 | [+2.2, +62.6] | 12/20 |
| actor_critic-sp+biased/policy_selection | single_sub_policy | +71.5 | [+41.8, +106.2] | 20/20 |
| actor_critic-sp+biased/policy_selection | best_fixed_sub_policy | +30.3 | [+20.3, +44.3] | 17/20 |
| actor_critic-sp+biased/policy_selection | oracle_best_sub_policy | -8.9 | [-21.1, +3.0] | 7/20 |
| actor_critic-sp+biased/policy_selection | mappo | +27.4 | [+20.3, +34.7] | 19/20 |
| actor_critic-sp+biased/policy_selection | traj_s36 | +41.9 | [+30.5, +54.3] | 20/20 |
| actor_critic-sp+biased/policy_selection | hsp_s36 | +11.4 | [-1.4, +23.0] | 14/20 |
| actor_critic-sp+biased/policy_selection | fcp_s36 | +27.4 | [+19.8, +37.2] | 18/20 |
| actor_critic-sp+biased/policy_selection | mep_s36 | +29.4 | [+19.6, +40.3] | 19/20 |
| actor_critic-sp+biased/policy_selection | mappo_argmax | +32.2 | [+24.5, +40.1] | 19/20 |
| actor_critic-sp+biased/policy_selection | traj_s36_argmax | +41.7 | [+31.3, +53.4] | 20/20 |
| actor_critic-sp+biased/policy_selection | hsp_s36_argmax | +5.0 | [-6.5, +15.6] | 11/20 |
| actor_critic-sp+biased/policy_selection | fcp_s36_argmax | +30.1 | [+21.7, +40.5] | 19/20 |
| actor_critic-sp+biased/policy_selection | mep_s36_argmax | +26.7 | [+14.5, +39.8] | 16/20 |
| actor_critic-sp+biased/learned_static_ensemble | static_ensemble | +12.4 | [+1.0, +25.8] | 13/20 |
| actor_critic-sp+biased/learned_static_ensemble | static_ensemble_argmax | -5.6 | [-20.9, +9.4] | 9/20 |
| actor_critic-sp+biased/learned_static_ensemble | single_sub_policy | +35.3 | [+19.0, +53.2] | 18/20 |
| actor_critic-sp+biased/learned_static_ensemble | best_fixed_sub_policy | -5.8 | [-18.1, +4.7] | 12/20 |
| actor_critic-sp+biased/learned_static_ensemble | oracle_best_sub_policy | -45.0 | [-59.6, -30.0] | 1/20 |
| actor_critic-sp+biased/learned_static_ensemble | mappo | -8.8 | [-24.8, +5.6] | 12/20 |
| actor_critic-sp+biased/learned_static_ensemble | traj_s36 | +5.8 | [-5.7, +16.8] | 15/20 |
| actor_critic-sp+biased/learned_static_ensemble | hsp_s36 | -24.8 | [-39.8, -9.4] | 3/20 |
| actor_critic-sp+biased/learned_static_ensemble | fcp_s36 | -8.7 | [-28.7, +9.7] | 11/20 |
| actor_critic-sp+biased/learned_static_ensemble | mep_s36 | -6.8 | [-30.7, +14.7] | 14/20 |
| actor_critic-sp+biased/learned_static_ensemble | mappo_argmax | -3.9 | [-19.2, +10.0] | 13/20 |
| actor_critic-sp+biased/learned_static_ensemble | traj_s36_argmax | +5.6 | [-7.7, +17.8] | 15/20 |
| actor_critic-sp+biased/learned_static_ensemble | hsp_s36_argmax | -31.2 | [-44.5, -17.9] | 3/20 |
| actor_critic-sp+biased/learned_static_ensemble | fcp_s36_argmax | -6.0 | [-27.7, +13.5] | 13/20 |
| actor_critic-sp+biased/learned_static_ensemble | mep_s36_argmax | -9.5 | [-35.4, +13.4] | 13/20 |

Saved to /cmlscratch/kpowell1/adhoc_teamwork/runs/random0_medium/hsp-w1/analysis
