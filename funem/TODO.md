# TODO

* Maybe allow some larger moves? - it might be hard for a bad policy to ever
see how to get closer when far away and noise might be swamping the currently
largest possible moves.
* Test a deeper network.
* Test the PPO algorithm for learning a policy.
* Implement a "fitter" policy - should be the real classical benchmark.
* Build a RNN to make a predictive model of the next set of sensor values given
an observation.
* Normalize the inputs? - observation should have zero mean?
* Do we need a LSTM to recall previous states?
* Test a larger observation sequence.
* Implement historical data sources.
* Add a tracker for integrated total heat over time.
* Implement a kick-start model based on using the `SimpleRuleBased` q-learner.
* Carefully double check training batch construction to make sure we are
doing Q-learning correctly.
* Make long runs use a running mean for the entries in the plots, etc.
    * Do this by making plot sequences a buffer of fixed size and every time
    they fill up, reduce them all by a factor of two in time and squeeze down
    to use only half the entries.
