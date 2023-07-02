
How to train a model that generalizes to an arbitrary domain with only the training samples, but not the corresponding domain information, as these domain information may not be available in the real world. Our paper builds upon this set-up and aims to offer a solution that allows the model to be robustly trained without domain information and to empirically perform well on unseen domains.   

This approach discards the representations associated with the higher gradients at each epoch, and forces the model to predict with remaining information.