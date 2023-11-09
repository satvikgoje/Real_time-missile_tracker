# Tracker-implementation
Used two types of trackers a correlation based and a frequency based tracker, using MMSE and implemented a pre-screener for extracting a set of features
in order to track the target when an occlusion is present.

# Changes introduced in the paper
+ The paper presented above implements a correlation based algorithm with very less specifications, our approach includes phase correlation
to get the position of the tank faster instead of mean squared error.

+ to improve processing time we also reduced the no.of features to track the centroids faster, in the feature based approach where occlusion was a problem.
