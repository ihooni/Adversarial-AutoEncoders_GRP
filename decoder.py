import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy.matlib as npm


class HyperSphereInsideSampler(object):
    """
    Sampler for prior z
    """
    def __init__(self, name='hyper_sphere_inside_sampler', r=3, z_dim=5):
        self.name = name
        self.r = r
        self.z_dim = z_dim

    def sample(self, n):
        z = np.random.randn(n, self.z_dim)
        z_norm = np.linalg.norm(z, axis=1)
        z_unit = z / npm.repmat(z_norm.reshape((-1, 1)), 1, self.z_dim)  # on the surface of a hypersphere
        u = np.power(np.random.rand(n, 1), (1 / self.z_dim) * np.ones(shape=(n, 1)))
        z_sphere = self.r * z_unit * npm.repmat(u, 1, self.z_dim)  # samples inside the hypersphere
        samples = z_sphere
        return samples

    def plot(self, n=1000, tfs=20):
        samples = self.sample(n=n)
        plt.figure(figsize=(6, 6))
        plt.plot(samples[:, 0], samples[:, 1], 'k.')
        plt.xlim(-self.r, self.r)
        plt.ylim(-self.r, self.r)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(self.name, fontsize=tfs)
        plt.show()


class AAEDecoder(nn.Module):
    def __init__(self, y_dim, z_dim, hidden_dim, x_hat_dim):
        super(AAEDecoder, self).__init__()
        self.input_y_block = nn.Sequential(
            nn.Linear(y_dim, 16),
            nn.LeakyReLU(0.2)
        )

        self.input_z_block = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(0.2)
        )

        self.out_block = nn.Sequential(
            nn.Linear(16 + hidden_dim, hidden_dim),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, x_hat_dim),
            nn.Tanh()
        )

    def forward(self, y, z):
        out_y = self.input_y_block(y)
        out_z = self.input_z_block(z)
        x_hat = self.out_block(torch.cat((out_y, out_z), dim=1))
        return x_hat


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    traj_len = 100
    x_dim = traj_len * 3
    hidden_dim = 512
    y_dim = 6
    z_dim = 100

    # change below values if you want
    model_path = './save_model/lr-scheduling-hyperspherev2-0.5/aae-decoder-6000.pt'
    # start = [0.040866, 0.040866, -0.27577937]
    # end = [0.12268186, -0.15862494,  0.19555572]
    start = [-0.1, -0.1, -0.1]
    end = [0.1, 0.1, 0.1]
    num_of_traj = 100
    ###

    z_sampler = HyperSphereInsideSampler(r=3, z_dim=z_dim)

    # load pretrained Decoder
    pre_P = AAEDecoder(y_dim, z_dim, hidden_dim, x_dim)
    pre_P.load_state_dict(torch.load(model_path))
    pre_P.to(device)

    y = np.repeat(np.concatenate((start, end), axis=None).reshape(1, -1), num_of_traj, axis=0)
    y = torch.FloatTensor(y).to(device)
    prior_z = torch.FloatTensor(z_sampler.sample(num_of_traj)).to(device)

    trajectories = pre_P(y, prior_z)

    trajectory = trajectories[0]
    print(trajectory)

    # trajectory = trajectory.reshape(3, 100).T.detach().cpu().numpy()
    # print(np.linalg.norm(trajectory[:99, :] - trajectory[1:, :], axis=1))

    # # plot result
    # plt.style.use('seaborn-whitegrid')
    # fig = plt.figure(figsize=(12, 10))
    # ax = fig.gca(projection='3d')
    #
    # trajectories = trajectories.cpu().detach().numpy()
    #
    # for traj in trajectories:
    #     ax.plot3D(traj[:100], traj[100:200], traj[200:])
    # plt.show()
