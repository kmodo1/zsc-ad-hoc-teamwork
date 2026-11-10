#!/bin/bash
# Re-run the analysis for every layout and population that has results, then copy
# summaries and the main plots into results/overcooked/. Submit from the repo root:
#   sbatch -J adhoc-analyze --gres=gpu:0 --ntasks=2 adhoc_teamwork/slurm/run.sbatch bash adhoc_teamwork/slurm/analyze_all.sh
R=$ADHOC_SCRATCH/runs; D=$ADHOC_REPO/adhoc_teamwork/results/overcooked
for layout in unident_s random0_medium random3; do
    for pop in traj-full10 traj-s10 s2-mix hsp-w1 hsp-w1-new traj-new10 mix-g8s4 mix-g4s8 mix-g4s4t4; do
        [ -f $R/$layout/$pop/population_eval/results.jsonl ] || continue
        python -m adhoc_teamwork.scripts.analyze_overcooked --root $R --layout $layout --population $pop > $R/$layout/$pop/analysis_output.md 2>&1 || { echo "FAILED $layout $pop"; tail -3 $R/$layout/$pop/analysis_output.md; continue; }
        mkdir -p $D/$layout/$pop
        cp $R/$layout/$pop/analysis/summary.json $R/$layout/$pop/analysis/method_comparison_biased.png $D/$layout/$pop/
        [ -f $R/$layout/$pop/analysis/population_steps.json ] && cp $R/$layout/$pop/analysis/population_steps.json $D/$layout/$pop/
        cp $R/$layout/$pop/analysis_output.md $D/$layout/$pop/tables.md
        # Weight plots only for the main runs and the memoryless comparison, to keep the repo small.
        for arm in actor_critic-sp actor_critic-sp+biased actor_critic-sp+biased-memoryless; do
            img=$R/$layout/$pop/analysis/weight_traces_$arm.png; [ -f "$img" ] && cp "$img" $D/$layout/$pop/
        done
        if [ -f $R/$layout/$pop/crossplay/matrix.npz ]; then cp $R/$layout/$pop/crossplay/crossplay_matrix.png $R/$layout/$pop/crossplay/self_play_vs_cross_play_gap.json $D/$layout/$pop/; fi
    done
done
du -sh $D
