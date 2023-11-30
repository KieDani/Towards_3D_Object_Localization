import torchvision.transforms.functional as tvF


class Normalize(object):
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        image = x['image']
        heatmap = x['heatmap'] if 'heatmap' in x else None
        image = tvF.normalize(image, self.mean, self.std)
        return {'image': image, 'heatmap': heatmap}



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class DeNormalize(object):
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        image = x['image']
        heatmap = x['heatmap'] if 'heatmap' in x else None
        image = tvF.normalize(image, [-m/s for m, s in zip(self.mean, self.std)], [1/s for s in self.std])
        image *= 255.
        return {'image': image, 'heatmap': heatmap}



train_transform = Compose([
    Normalize(),
])

val_transform = Compose([
    Normalize(),
])

denorm = DeNormalize()


if __name__ == '__main__':
    import torch
    norm = Normalize()
    denorm = DeNormalize()
    x = torch.rand(3, 4, 4)
    y = denorm(norm({'image': x / 255.}))['image']
    print(x)
    print(y)
    print((x - y) / x)