# TODO

* Implement target networks.
* Maybe allow some larger moves? - it might be hard for a bad policy to ever
see how to get closer when far away and noise might be swamping the currently
largest possible moves.
* Test a deeper network.
* Test a larger observation sequence.
* Implement linear annealing - should be using model 95% of the time after 10k
steps and then 99% of the time after another 10k steps?
* Implement historical data sources.
* Add a tracker for integrated total heat over time.
* Implement a kick-start model based on using the `SimpleRuleBased` q-learner.
* Carefully double check training batch construction to make sure we are
doing Q-learning correctly.
* Make long runs use a running mean for the entries in the plots, etc.
