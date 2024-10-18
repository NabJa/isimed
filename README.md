# ISImed
## A Framework for Self-Supervised Learning using Intrinsic Spatial Information in Medical Images


ISImed, is based on the observation that medical images exhibit a much lower variability among different images compared to classic data vision benchmarks.  By leveraging this resemblance of human body structures across multiple images, we establish a self-supervised objective that creates a latent representation capable of capturing its location in the physical realm. More specifically, our method involves sampling image crops and creating a distance matrix that compares the learned representation vectors of all possible combinations of these crops to the true distance between them. The intuition is, that the learned latent space is a positional encoding for a given image crop. 
