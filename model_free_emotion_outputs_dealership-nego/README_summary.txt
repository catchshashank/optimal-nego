Pipeline completed.

Input file:
/content/stage1_turns_dealership-nego.csv

Detected:
- 48 conversations
- 2480 turn-level rows
- 27 emotion/SST columns
- outcomes: ['outcome_binary']

Main outputs:
1. 01_uncertainty_trajectories
   Confidence bands for emotional trajectories and buyer-seller gaps.

2. 02_model_free_features
   Conversation-level summaries: mean, std, first, last, delta, min, max,
   early mean, late mean, and buyer-seller emotional gaps.

3. 03_outcome_validation
   Model-free correlations and simple predictive models linking emotions to outcomes.

4. 04_temporal_predictability
   Prediction-over-time curves showing when emotional signals become useful.

5. 05_frequency_vs_predictiveness
   Tables comparing emotion prevalence against predictive importance.

Recommended next step:
Inspect confidence bands and model-free correlations before interpreting
state-space or reinforcement-learning results.