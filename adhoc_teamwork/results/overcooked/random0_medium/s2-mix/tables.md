
## random0_medium / s2-mix / partners: biased (20)

| method | mean return | std across seeds | seeds |
|---|---:|---:|---:|
| actor_critic-sp+biased/adaptive_ensemble_argmax | 162.3 | 0.7 | 3 |
| oracle_best_sub_policy | 156.7 | 0.0 | 1 |
| static_ensemble_argmax | 156.4 | 0.0 | 1 |
| actor_critic-sp+biased/policy_selection | 151.0 | 1.7 | 3 |
| best_fixed_sub_policy | 148.9 | 0.0 | 1 |
| hsp_s36_argmax | 140.8 | 14.9 | 5 |
| actor_critic-sp+biased/adaptive_ensemble | 140.6 | 0.9 | 3 |
| actor_critic-sp+biased/top_learned_sub_policy | 137.3 | 11.1 | 3 |
| hsp_s36 | 134.3 | 11.1 | 5 |
| mep_s36_argmax | 119.1 | 10.2 | 5 |
| mappo | 118.4 | 14.1 | 5 |
| fcp_s36 | 118.3 | 11.0 | 5 |
| actor_critic-sp+biased/learned_static_ensemble | 116.8 | 1.5 | 3 |
| mep_s36 | 116.3 | 12.2 | 5 |
| single_sub_policy | 115.7 | 20.6 | 12 |
| fcp_s36_argmax | 115.6 | 13.0 | 5 |
| mappo_argmax | 113.5 | 17.9 | 5 |
| static_ensemble | 112.5 | 0.0 | 1 |
| traj_s36_argmax | 104.0 | 19.9 | 5 |
| traj_s36 | 103.8 | 21.2 | 5 |

## Differences on biased partners (ours - reference, 95% bootstrap CI over distinct partner agents)

| ours | reference | difference | 95% CI | partners better |
|---|---|---:|---|---:|
| actor_critic-sp+biased/adaptive_ensemble | actor_critic-sp+biased/learned_static_ensemble | +23.8 | [+18.6, +29.0] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble | actor_critic-sp+biased/top_learned_sub_policy | +3.4 | [+0.8, +6.2] | 15/20 |
| actor_critic-sp+biased/adaptive_ensemble | static_ensemble | +28.1 | [+22.4, +34.3] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble | static_ensemble_argmax | -15.8 | [-24.9, -6.1] | 5/20 |
| actor_critic-sp+biased/adaptive_ensemble | single_sub_policy | +24.9 | [+22.2, +27.4] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble | best_fixed_sub_policy | -8.3 | [-15.7, -1.7] | 6/20 |
| actor_critic-sp+biased/adaptive_ensemble | oracle_best_sub_policy | -16.1 | [-20.9, -12.1] | 1/20 |
| actor_critic-sp+biased/adaptive_ensemble | mappo | +22.3 | [+14.1, +29.7] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble | traj_s36 | +36.8 | [+28.9, +44.4] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble | hsp_s36 | +6.3 | [-1.1, +13.2] | 13/20 |
| actor_critic-sp+biased/adaptive_ensemble | fcp_s36 | +22.3 | [+15.5, +29.1] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble | mep_s36 | +24.3 | [+13.9, +33.9] | 16/20 |
| actor_critic-sp+biased/adaptive_ensemble | mappo_argmax | +27.1 | [+18.4, +34.6] | 19/20 |
| actor_critic-sp+biased/adaptive_ensemble | traj_s36_argmax | +36.6 | [+28.8, +44.2] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble | hsp_s36_argmax | -0.1 | [-6.0, +5.3] | 8/20 |
| actor_critic-sp+biased/adaptive_ensemble | fcp_s36_argmax | +25.0 | [+16.3, +33.6] | 17/20 |
| actor_critic-sp+biased/adaptive_ensemble | mep_s36_argmax | +21.6 | [+9.3, +32.9] | 16/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | actor_critic-sp+biased/learned_static_ensemble | +45.5 | [+36.7, +53.5] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | actor_critic-sp+biased/top_learned_sub_policy | +25.1 | [+18.1, +32.4] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | static_ensemble | +49.8 | [+40.5, +59.1] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | static_ensemble_argmax | +6.0 | [+0.8, +11.4] | 11/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | single_sub_policy | +46.6 | [+39.5, +53.1] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | best_fixed_sub_policy | +13.4 | [+1.3, +23.6] | 15/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | oracle_best_sub_policy | +5.6 | [-3.9, +14.5] | 14/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mappo | +44.0 | [+37.2, +49.7] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | traj_s36 | +58.5 | [+49.2, +67.2] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | hsp_s36 | +28.0 | [+15.7, +39.3] | 18/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | fcp_s36 | +44.0 | [+36.7, +51.7] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mep_s36 | +46.0 | [+35.7, +56.2] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mappo_argmax | +48.8 | [+40.9, +55.7] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | traj_s36_argmax | +58.4 | [+50.5, +66.0] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | hsp_s36_argmax | +21.6 | [+11.1, +30.8] | 16/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | fcp_s36_argmax | +46.7 | [+38.1, +55.4] | 20/20 |
| actor_critic-sp+biased/adaptive_ensemble_argmax | mep_s36_argmax | +43.3 | [+31.5, +55.5] | 20/20 |
| actor_critic-sp+biased/policy_selection | actor_critic-sp+biased/learned_static_ensemble | +34.2 | [+26.9, +41.5] | 20/20 |
| actor_critic-sp+biased/policy_selection | actor_critic-sp+biased/top_learned_sub_policy | +13.8 | [+11.3, +16.2] | 20/20 |
| actor_critic-sp+biased/policy_selection | static_ensemble | +38.5 | [+30.7, +46.6] | 20/20 |
| actor_critic-sp+biased/policy_selection | static_ensemble_argmax | -5.3 | [-14.5, +3.8] | 7/20 |
| actor_critic-sp+biased/policy_selection | single_sub_policy | +35.4 | [+31.2, +39.2] | 20/20 |
| actor_critic-sp+biased/policy_selection | best_fixed_sub_policy | +2.1 | [-6.8, +10.4] | 13/20 |
| actor_critic-sp+biased/policy_selection | oracle_best_sub_policy | -5.7 | [-11.3, -1.0] | 8/20 |
| actor_critic-sp+biased/policy_selection | mappo | +32.7 | [+26.1, +39.2] | 20/20 |
| actor_critic-sp+biased/policy_selection | traj_s36 | +47.2 | [+37.7, +56.7] | 20/20 |
| actor_critic-sp+biased/policy_selection | hsp_s36 | +16.7 | [+8.3, +24.4] | 17/20 |
| actor_critic-sp+biased/policy_selection | fcp_s36 | +32.7 | [+26.7, +39.3] | 20/20 |
| actor_critic-sp+biased/policy_selection | mep_s36 | +34.7 | [+25.8, +43.2] | 20/20 |
| actor_critic-sp+biased/policy_selection | mappo_argmax | +37.5 | [+30.2, +44.4] | 20/20 |
| actor_critic-sp+biased/policy_selection | traj_s36_argmax | +47.1 | [+37.6, +56.2] | 20/20 |
| actor_critic-sp+biased/policy_selection | hsp_s36_argmax | +10.3 | [+3.2, +16.9] | 14/20 |
| actor_critic-sp+biased/policy_selection | fcp_s36_argmax | +35.4 | [+27.6, +43.6] | 19/20 |
| actor_critic-sp+biased/policy_selection | mep_s36_argmax | +32.0 | [+21.0, +42.3] | 20/20 |
| actor_critic-sp+biased/learned_static_ensemble | static_ensemble | +4.3 | [+1.9, +6.9] | 16/20 |
| actor_critic-sp+biased/learned_static_ensemble | static_ensemble_argmax | -39.5 | [-50.6, -27.0] | 0/20 |
| actor_critic-sp+biased/learned_static_ensemble | single_sub_policy | +1.2 | [-4.0, +5.7] | 14/20 |
| actor_critic-sp+biased/learned_static_ensemble | best_fixed_sub_policy | -32.1 | [-38.5, -26.3] | 0/20 |
| actor_critic-sp+biased/learned_static_ensemble | oracle_best_sub_policy | -39.9 | [-47.3, -32.1] | 0/20 |
| actor_critic-sp+biased/learned_static_ensemble | mappo | -1.5 | [-12.7, +8.6] | 11/20 |
| actor_critic-sp+biased/learned_static_ensemble | traj_s36 | +13.0 | [+8.8, +16.8] | 19/20 |
| actor_critic-sp+biased/learned_static_ensemble | hsp_s36 | -17.5 | [-24.8, -9.5] | 2/20 |
| actor_critic-sp+biased/learned_static_ensemble | fcp_s36 | -1.4 | [-12.3, +8.0] | 12/20 |
| actor_critic-sp+biased/learned_static_ensemble | mep_s36 | +0.5 | [-14.1, +13.1] | 14/20 |
| actor_critic-sp+biased/learned_static_ensemble | mappo_argmax | +3.3 | [-8.2, +13.0] | 13/20 |
| actor_critic-sp+biased/learned_static_ensemble | traj_s36_argmax | +12.9 | [+7.5, +17.9] | 18/20 |
| actor_critic-sp+biased/learned_static_ensemble | hsp_s36_argmax | -23.9 | [-29.4, -18.3] | 2/20 |
| actor_critic-sp+biased/learned_static_ensemble | fcp_s36_argmax | +1.2 | [-11.3, +12.7] | 11/20 |
| actor_critic-sp+biased/learned_static_ensemble | mep_s36_argmax | -2.2 | [-18.6, +12.3] | 11/20 |

Saved to /cmlscratch/kpowell1/adhoc_teamwork/runs/random0_medium/s2-mix/analysis
