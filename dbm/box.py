import numpy as np

class Box():

    def __init__(self, file):

        self.dim = self.get_box_dim(file)
        self.dim_inv= np.linalg.inv(self.dim)
        self.v1 = self.dim[:, 0]
        self.v2 = self.dim[:, 1]
        self.v3 = self.dim[:, 2]
        self.volume = self.get_vol()
        self.center = 0.5*self.v1 + 0.5*self.v2 + 0.5*self.v3

    def get_box_dim(self, file):
        # reads the box dimensions from the last line in the gro file
        f_read = open(file, "r")
        bd = np.array(f_read.readlines()[-1].split(), np.float32)
        f_read.close()
        bd = list(bd)
        for n in range(len(bd), 10):
            bd.append(0.0)
        dim = np.array([[bd[0], bd[5], bd[7]],
                                 [bd[3], bd[1], bd[8]],
                                 [bd[4], bd[6], bd[2]]])
        return dim


    def move_inside(self, pos):
        f = np.dot(self.dim_inv, pos)
        g = f - np.floor(f)
        new_pos = np.dot(self.dim, g)
        return new_pos

    def diff_vec(self, diff_vec):
        diff_vec = diff_vec + self.center
        diff_vec = self.move_inside(diff_vec)
        diff_vec = diff_vec - self.center
        return diff_vec

    def diff_vec_batch(self, diff_vec):
        diff_vec = np.swapaxes(diff_vec, 0, 1)
        diff_vec = diff_vec + self.center[:, np.newaxis]
        diff_vec = self.move_inside(diff_vec)
        diff_vec = diff_vec - self.center[:, np.newaxis]
        diff_vec = np.swapaxes(diff_vec, 0, 1)
        return diff_vec

    def get_vol(self):
        norm1 = np.sqrt(np.sum(np.square(self.v1)))
        norm2 = np.sqrt(np.sum(np.square(self.v2)))
        norm3 = np.sqrt(np.sum(np.square(self.v3)))

        cos1 = np.sum(self.v2 * self.v3) / (norm2 * norm3)
        cos2 = np.sum(self.v1 * self.v3) / (norm1 * norm3)
        cos3 = np.sum(self.v1 * self.v2) / (norm1 * norm2)
        v = norm1*norm2*norm3 * np.sqrt(1-np.square(cos1)-np.square(cos2)-np.square(cos3)+2*np.sqrt(cos1*cos2*cos3))
        return v


