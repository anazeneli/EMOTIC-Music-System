# takes in features extracted from person_feature_extractor and
# global_feature_extractor combines them in a separate fusion network

# uses global average pooling layer on each feature map
# to reduce the number of features

# fully connected layer reduces dimensionality of concatenated
# pooled features -- output is a 256-D vector

# large fully connected layer to allow for training process
# to learn independent representations for each task
# split into two branches: continuous dimensions (3) and discrete categories(26)
# batch normalization and ReLU added after each convolutional layer
# parameters of three modules learned jointly using
# stochastic gradient descent with momentum
# batch size = 52 (2x num categories)
# use uniform sampling per category to have at least one instance of each
# discrete category

# loss is defined as the individual losses of discrete and continuous
# N = 26
# l_disc = 1/N * sum 1..N(wi(y^i_disc - yi_disc)^2)
# wi = 1/ln(c+pi)]


# C = {Valence, Arousal, Dominance}
# l_disc = 1/C * sum 1..C(vk(y^k_cont - yk_cont)^2)

#l_comb = lam_d * l_disc + lam_cont * l_cont

class FusionModule():
    def __init__(self, bodies, context):
        self.models = [bodies, context]

