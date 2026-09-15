from geomloss import SamplesLoss


def compute_wasserstein(real_samples, generated_samples, p=1, blur=0.05,):

    loss = SamplesLoss(loss="sinkhorn", p=p, blur=blur)

    distance = loss(real_samples, generated_samples)

    return distance.item()