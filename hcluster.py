from itertools import combinations
import numpy as np
from PIL import Image, ImageDraw
import os


class ClusterNode:
    def __init__(
        self,
        vec: np.ndarray,
        left,
        right,
        distance=0.0,
        count=1
    ):
        self.vec = vec
        self.left = left
        self.right = right
        self.distance = distance
        self.count = count
    
    def extract_clusters(
        self,
        dist
    ):
        if self.distance < dist:
            return [self]
        return self.left.extract_clusters(dist) + self.right.extract_clusters(dist)
    
    def get_cluster_elements(
        self
    ):
        return self.left.get_cluster_elements() + self.right.get_cluster_elements()
    
    def get_height(
        self
    ):
        return self.left.get_height() + self.right.get_height()
    
    def get_depth(
        self
    ):
        return max(self.left.get_depth(), self.right.get_depth()) + self.distance
    
    def draw(
        self,
        draw,
        x,
        y,
        scale,
        imlist,
        im
    ):
        h1 = int(self.left.get_height() * 20 /2)
        h2 = int(self.right.get_height() * 20 /2)
        top = y - (h1 + h2)
        bottom = y + (h1 + h2)
        draw.line((x, top + h1, x, bottom - h2), fill=(0, 0, 0))
        ll = self.distance * scale
        draw.line((x, top + h1, x + ll, top + h1), fill=(0, 0, 0))
        draw.line((x, bottom - h2, x + ll, bottom - h2), fill=(0, 0, 0))

        self.left.draw(draw, x + ll, top + h1, scale, imlist, im)
        self.right.draw(draw, x + ll, bottom - h2, scale, imlist, im)


class ClusterLeafNode:
    def __init__(
        self,
        vec: np.ndarray,
        id
    ):
        self.vec = vec
        self.id = id
    
    def extract_clusters(
        self,
        dist
    ):
        return [self]
    
    def get_cluster_elements(
        self
    ):
        return [self.id]
    
    def get_height(
        self
    ):
        return 1
    
    def get_depth(
        self
    ):
        return 0
    
    def draw(
        self,
        draw,
        x,
        y,
        scale,
        imlist,
        im
    ):
        nodeim = Image.open(imlist[self.id])
        nodeim.thumbnail([20, 20])
        ns = nodeim.size
        im.paste(
            nodeim,
            [int(x), int(y - ns[1] // 2), int(x + ns[0]), int(y + ns[1] - ns[1] // 2)]
        )


def l2_dist(
    v1: np.ndarray,
    v2: np.ndarray
):
    return np.sqrt(np.sum((v1 - v2)**2))


def l1_dist(
    v1: np.ndarray,
    v2: np.ndarray
):
    return np.sum(np.abs(v1 - v2))


def hcluster(
    features,
    dist_func=l2_dist
):
    distances = {}
    node = [ClusterLeafNode(np.array(f), id=i) for i, f in enumerate(features)]

    while len(node) > 1:
        closest = float('Inf')

        for ni, nj in combinations(node, 2):
            if (ni, nj) not in distances:
                distances[ni, nj] = dist_func(ni.vec, nj.vec)
            
            d = distances[ni, nj]
            if d < closest:
                closest = d
                lowestpair = (ni, nj)
        ni, nj = lowestpair

        new_vec = (ni.vec + nj.vec) / 2.0

        new_node = ClusterNode(
            new_vec,
            left=ni,
            right=nj,
            distance=closest
        )
        node.remove(ni)
        node.remove(nj)
        node.append(new_node)
    
    return node[0]


def draw_dendrogram(
    node,
    imlist,
    filename=os.path.join('images_output', 'dendrogram_clusters.jpg')
):
    rows = node.get_height() * 20
    cols = 1200

    scale = float(cols-150) / node.get_depth()

    im = Image.new('RGB', (cols, rows), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    draw.line((0, rows / 2, 20, rows / 2), fill=(0, 0, 0))
    node.draw(draw, 20, (rows / 2), scale, imlist, im)
    im.save(filename)
    im.show()
